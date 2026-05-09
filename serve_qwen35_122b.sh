#!/usr/bin/env bash
# Launch SGLang serving Qwen/Qwen3.5-122B-A10B-FP8 on 2× B200, TP=2, port 30002.
#
# Qwen3.5-122B-A10B is a 122B-total / 10B-active MoE (256 experts, 8 per token,
# 32 attention heads). FP8 weights ~120 GB → fits 2× B200 TP=2 with comfortable
# headroom. TP=2 head-divisibility: 32/2=16 ✓; KV heads 2/2=1 ✓; experts 256/2=128 ✓.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/serve_qwen35_122b.log"
PIDFILE="$REPO_ROOT/serve_qwen35_122b.pid"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30002}"
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
READY_DEADLINE_SEC="${READY_DEADLINE_SEC:-2400}"   # 40 min — covers weight download + warmup

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

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')
if [[ "${USED_MIB:-0}" -gt 30000 ]]; then
    echo "[serve] WARN: GPUs $GPUS hold ${USED_MIB} MiB total — vacate them first or expect OOM."
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:${PATH}"
LIBNUMA_LIB_PATH="$REPO_ROOT/.build/libnuma/usr/lib/x86_64-linux-gnu"
CONDA_LIB_PATH="/opt/conda/lib"
if [[ ! -f "$LIBNUMA_LIB_PATH/libnuma.so.1" ]]; then
    echo "[serve] FAIL: libnuma not extracted at $LIBNUMA_LIB_PATH. Re-run install_v4_flash.sh."
    exit 1
fi
export LD_LIBRARY_PATH="$CONDA_LIB_PATH:$LIBNUMA_LIB_PATH:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "[serve] $(date -Iseconds) starting SGLang"
echo "[serve]   model        = $MODEL"
echo "[serve]   tp           = 2"
echo "[serve]   gpus         = $CUDA_VISIBLE_DEVICES"
echo "[serve]   host:port    = $HOST:$PORT"
echo "[serve]   mem_static   = $MEM_FRACTION_STATIC"
echo "[serve]   ready deadln = ${READY_DEADLINE_SEC}s"
echo "[serve]   log          = $LOG"
echo "[serve]   pid file     = $PIDFILE"

nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp 2 \
    --host "$HOST" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --chunked-prefill-size 4096 \
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
