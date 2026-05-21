#!/usr/bin/env bash
# Launch SGLang serving Kimi-K2.6 on GPUs 0-3, TP=4, port 30000.
#
# Kimi-K2.6 is a 1T-param MoE, native INT4 (compressed-tensors, weight-only W4A16).
# Weights ~594 GB live in /asmp-models/huggingface-cache (the default ~/.cache on
# /home has only ~422 GB free), so HF_HOME is pointed there. On B200 the INT4
# weights run W4A16 — dequantized to BF16 in-kernel (no INT4 tensor cores);
# lossless precision, ~2x slower than FP4 in the compute-bound regime.
#
# ~594 GB across 4x 179 GiB is a tight fit: weights alone are ~0.77 of HBM, so
# mem-fraction-static is 0.90 (leaves ~23 GiB/GPU for the KV pool). This is a
# first-launch estimate -- if it OOMs on weight load lower it, if it OOMs on KV
# or graph capture adjust accordingly; the log tail on failure shows which.
#
# Run AFTER the K2.6 download finishes and AFTER vacating GPUs 0-3 (stop the
# gemma-4-31B / gpt-oss-120b / Qwen3.5-122B servers). Refuses to start if port
# 30000 is already taken.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/serve_kimi_k2_6.log"
PIDFILE="$REPO_ROOT/serve_kimi_k2_6.pid"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-moonshotai/Kimi-K2.6}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
READY_DEADLINE_SEC="${READY_DEADLINE_SEC:-1800}"   # 30 min covers 594 GB weight load + warmup

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

# Warn (don't block) if GPUs 0-3 still hold memory — the gemma/gpt-oss/Qwen3.5-122B
# servers may not be stopped yet. K2.6 will OOM on weight load if they are resident.
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')
if [[ "${USED_MIB:-0}" -gt 30000 ]]; then
    echo "[serve] WARN: GPUs $GPUS hold ${USED_MIB} MiB total — another server may still be resident."
    echo "        Server will likely OOM on weight load. Stop the gemma/gpt-oss/Qwen3.5-122B servers first."
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
# K2.6 weights live on /asmp-models — the default ~/.cache on /home cannot hold them.
export HF_HOME="${HF_HOME:-/asmp-models/huggingface-cache}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
# nvcc must be reachable for any first-run JIT recompile of flashinfer-jit-cache.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:${PATH}"
# libnuma.so.1 is required by sgl_kernel/sm100/common_ops.abi3.so but not on the
# host; install_v4_flash.sh extracts it to .build/libnuma/. /opt/conda/lib provides
# a libstdc++.so.6 with CXXABI_1.3.15 (system /usr/lib only has up to 1.3.13).
# Order: conda first (libstdc++), then libnuma, then cuda runtime.
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
echo "[serve]   hf_home      = $HF_HOME"
echo "[serve]   mem_static   = $MEM_FRACTION_STATIC"
echo "[serve]   ready deadln = ${READY_DEADLINE_SEC}s"
echo "[serve]   log          = $LOG"
echo "[serve]   pid file     = $PIDFILE"

# Launch detached. nohup keeps it alive past this shell; setsid makes it a new
# session so kill -TERM <pid> reaches the whole process tree.
# --disable-custom-all-reduce: SGLang's custom P2P/IPC all-reduce fails CUDA
# graph capture in this k8s container (custom_all_reduce.cuh: CUDA invalid
# argument). Fall back to NCCL — same as the running Qwen3.5-122B TP2 server.
nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp 4 \
    --host "$HOST" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --chunked-prefill-size 4096 \
    --disable-custom-all-reduce \
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
    # Progress: print last few log lines once a minute so the user can watch warmup.
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
