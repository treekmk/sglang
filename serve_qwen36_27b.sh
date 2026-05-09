#!/usr/bin/env bash
# Launch SGLang serving Qwen/Qwen3.6-27B on a single B200, TP=1, port 30004.
#
# Qwen3.6-27B uses Qwen3_5ForConditionalGeneration (multimodal: vision encoder
# included, no audio). 27B BF16 ~54 GB fits 1× B200 with ~125 GB headroom for KV.
# 24 attention heads, 4 KV heads, head_dim 256. The vision tower stays idle for
# text-only /v1/score traffic — no extra config needed.
#
# This replaces the running vLLM on port 10004 once the user kills it. SGLang
# binds a different port (30004) so the migration can run side-by-side.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/serve_qwen36_27b.log"
PIDFILE="$REPO_ROOT/serve_qwen36_27b.pid"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30004}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
GPUS="${CUDA_VISIBLE_DEVICES:-4}"
READY_DEADLINE_SEC="${READY_DEADLINE_SEC:-1200}"   # 20 min — small dense model

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
    echo "[serve] WARN: GPU $GPUS holds ${USED_MIB} MiB total — vacate it first or expect OOM."
    echo "        (the running vLLM on port 10004 needs to be killed before launching this)"
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
echo "[serve]   tp           = 1"
echo "[serve]   gpus         = $CUDA_VISIBLE_DEVICES"
echo "[serve]   host:port    = $HOST:$PORT"
echo "[serve]   mem_static   = $MEM_FRACTION_STATIC"
echo "[serve]   ready deadln = ${READY_DEADLINE_SEC}s"
echo "[serve]   log          = $LOG"
echo "[serve]   pid file     = $PIDFILE"

nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp 1 \
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
