#!/usr/bin/env bash
# Launch SGLang serving DeepSeek-V4-Flash on GPUs 0-3, TP=4, port 30000.
#
# Recipe: B200 Low-Latency from test/manual/dsv4/test_b200_flash.py with MTP retained
#         (--speculative-* flags) and --mem-fraction-static 0.85 from the deploy spec.
#
# Run AFTER `bash install_v4_flash.sh` has succeeded and AFTER vacating GPUs 0-3
# (kill the vLLM on port 10000). This script does NOT touch port 10000 and refuses
# to start if port 30000 is already taken.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/serve_v4_flash.log"
PIDFILE="$REPO_ROOT/serve_v4_flash.pid"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
READY_DEADLINE_SEC="${READY_DEADLINE_SEC:-3600}"   # 60 min covers weights download + DeepGEMM warmup

# ---- Preflight ----------------------------------------------------------
if [[ ! -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    echo "[serve] FAIL: $REPO_ROOT/.venv missing. Run 'bash install_v4_flash.sh' first."
    exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PORT}\$"; then
    echo "[serve] FAIL: port $PORT already in use."
    exit 1
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[serve] FAIL: server already running with pid $(cat "$PIDFILE"). Stop it first:"
    echo "        kill -TERM \$(cat $PIDFILE)"
    exit 1
fi

# Warn (don't block) if GPUs 0-3 still hold a lot of memory — vLLM probably still resident.
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')
if [[ "${USED_MIB:-0}" -gt 30000 ]]; then
    echo "[serve] WARN: GPUs $GPUS hold ${USED_MIB} MiB total — vLLM may still be resident."
    echo "        Server will likely OOM on weight load. Kill the vLLM on port 10000 first:"
    echo "          ss -ltnp | grep ':10000' to find the pid; then kill it."
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
# nvcc must be reachable for any first-run JIT recompile of flashinfer-jit-cache (cu129 -> cu130)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:${PATH}"
# libnuma.so.1 is required by sgl_kernel/sm100/common_ops.abi3.so but not on the host;
# install_v4_flash.sh extracts it to .build/libnuma/. /opt/conda/lib provides a
# libstdc++.so.6 with CXXABI_1.3.15 (the system /usr/lib one only has up to 1.3.13);
# conda-shipped libicui18n.so.78 needs the newer ABI. Order: conda first (libstdc++),
# then libnuma, then cuda runtime.
LIBNUMA_LIB_PATH="$REPO_ROOT/.build/libnuma/usr/lib/x86_64-linux-gnu"
CONDA_LIB_PATH="/opt/conda/lib"
if [[ ! -f "$LIBNUMA_LIB_PATH/libnuma.so.1" ]]; then
    echo "[serve] FAIL: libnuma not extracted at $LIBNUMA_LIB_PATH. Re-run install_v4_flash.sh."
    exit 1
fi
export LD_LIBRARY_PATH="$CONDA_LIB_PATH:$LIBNUMA_LIB_PATH:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "[serve] $(date -Iseconds) starting SGLang"
echo "[serve]   model        = $MODEL"
echo "[serve]   tp           = 4"
echo "[serve]   gpus         = $CUDA_VISIBLE_DEVICES"
echo "[serve]   host:port    = $HOST:$PORT"
echo "[serve]   mem_static   = $MEM_FRACTION_STATIC"
echo "[serve]   ready deadln = ${READY_DEADLINE_SEC}s"
echo "[serve]   log          = $LOG"
echo "[serve]   pid file     = $PIDFILE"

# Launch detached. nohup keeps it alive past this shell; setsid makes it a new session
# so kill -TERM <pid> reaches the whole process tree.
nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp 4 \
    --host "$HOST" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "[serve] launched pid $SERVER_PID. Waiting for /v1/models (deadline ${READY_DEADLINE_SEC}s)..."

deadline=$(( $(date +%s) + READY_DEADLINE_SEC ))
last_tail_at=0
while (( $(date +%s) < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[serve] FAIL: server process exited early. Tail of $LOG:"
        tail -80 "$LOG"
        exit 1
    fi
    if curl -fsS --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[serve] $(date -Iseconds) READY. /v1/models returns 200."
        echo "[serve] To stop: kill -TERM \$(cat $PIDFILE)"
        exit 0
    fi
    # Progress: print last few log lines once a minute so the user can watch download/warmup
    now=$(date +%s)
    if (( now - last_tail_at >= 60 )); then
        echo "[serve] ($(date -Iseconds)) waiting; recent log:"
        tail -3 "$LOG" | sed 's/^/[serve]   /'
        last_tail_at=$now
    fi
    sleep 5
done

echo "[serve] FAIL: deadline (${READY_DEADLINE_SEC}s) exceeded; server not ready."
echo "[serve] Tail of $LOG:"
tail -80 "$LOG"
exit 1
