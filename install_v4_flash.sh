#!/usr/bin/env bash
# Install SGLang for DeepSeek-V4-Flash serving on B200 (sm_100), TP=4.
#
# Pins: origin/main @ 47e9ec11ad7a12b85aff4ed6efc5eb3a43961d16 (post-PR #23882
# "Deepseek V4" — V4 Flash + Pro support is now on main).
#
# Uses uv add (declarative; tracks deps in pyproject.toml + uv.lock at repo root).
# Most package versions are pinned by main's own python/pyproject.toml — we only
# need to add flashinfer-jit-cache (not declared by sglang itself) and the two
# from-source kernels (FlashMLA, DeepGEMM).
#
# Pinned versions (sourced from main's python/pyproject.toml):
#   torch:                2.11.0  (+cu130 wheel matches host nvcc cu13.0 — no
#                                   sitecustomize.py monkey-patch needed)
#   sglang-kernel:        0.4.2.post1
#   flashinfer_python:    0.6.8.post1
#   flashinfer_cubin:     0.6.8.post1
#   flashinfer-jit-cache: 0.6.8.post1 (cu130 wheel; added separately)
#   tilelang:             0.1.8 (in pyproject; no explicit add)
#   apache-tvm-ffi:       0.1.9 (in pyproject; no explicit add or re-pin)
#   sgl-deep-gemm:        0.0.1 (prebuilt stub; the real `deep_gemm` module
#                                comes from the from-source build below)
#   FlashMLA:             deepseek-ai/FlashMLA HEAD (cu130 + torch 2.11)
#   DeepGEMM:             sgl-project/DeepGEMM branch release-0426 @ 7f2a70
#
# Recipe origin: docker/deepseek_v4_b200.Dockerfile on origin/deepseek_v4.
# Deviations from the Dockerfile:
#   1. Pin main, not deepseek_v4. PR #23882 merged V4 Flash + Pro to main on
#      2026-05-07; main carries the same model files plus all the unrelated
#      improvements that have landed since the V4 branch diverged.
#   2. Host CUDA toolkit is 13.0 (Dockerfile assumes 12.9). cu13's libcudacxx
#      sits under /usr/local/cuda-13.0/include/cccl/ (cuda-cccl-13-0 package);
#      adding that to CPATH makes <cuda/std/...> resolve.
#   3. SGLang is added BEFORE FlashMLA. The pyproject pins torch==2.11.0; if
#      FlashMLA is built first, pip's resolver pulls a transient newer torch
#      and sglang then downgrades, leaving FlashMLA's .so with an ABI mismatch.
#   4. DeepGEMM is cloned from branch release-0426 (Dockerfile says 'release',
#      which no longer exists in the upstream repo).
#   5. libnuma1 is extracted to .build/libnuma/ (apt-get download, no sudo)
#      because the host doesn't ship libnuma.so.1 and sgl_kernel's prebuilt
#      sm100/common_ops.abi3.so dlopen()s it at import time.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- Pinned versions ----------------------------------------------------
SGLANG_PIN="47e9ec11ad7a12b85aff4ed6efc5eb3a43961d16"   # origin/main @ 2026-05-08
DEEPGEMM_PIN="7f2a70"
DEEPGEMM_BRANCH="release-0426"
FLASHINFER_JIT_CACHE_VER="0.6.8.post1"
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu130"
PYTHON_VER="3.11"

# ---- Toolchain ----------------------------------------------------------
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [[ -d "$CUDA_HOME/include/cccl" ]]; then
    export CPATH="$CUDA_HOME/include/cccl:${CPATH:-}"
fi

# Add user-local rust toolchain to PATH. SGLang's build-system requires
# setuptools-rust (or a transitive dep needs rustc) — without it, `uv add
# --editable ./python` errors with "can't find Rust compiler".
if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v rustc >/dev/null 2>&1; then
    echo "[install] rustc not on PATH; bootstrapping rustup to user dir (no sudo)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Bootstrap protoc to .build/protoc/. The sglang-grpc Rust crate's build.rs uses
# tonic-build / prost-build, which needs the protoc binary at compile time.
# Required since main: V4 branch had a Python-side proto compilation; main does
# it in Rust.
PROTOC_DIR="$REPO_ROOT/.build/protoc"
PROTOC_VER="29.3"
if [[ ! -x "$PROTOC_DIR/bin/protoc" ]]; then
    echo "[install] protoc not found; downloading $PROTOC_VER release binary to $PROTOC_DIR"
    mkdir -p "$PROTOC_DIR"
    curl -sLo "$PROTOC_DIR/protoc.zip" \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VER}/protoc-${PROTOC_VER}-linux-x86_64.zip"
    (cd "$PROTOC_DIR" && unzip -q -o protoc.zip)
fi
export PATH="$PROTOC_DIR/bin:$PATH"
export PROTOC="$PROTOC_DIR/bin/protoc"

command -v nvcc   >/dev/null 2>&1 || { echo "[install] FAIL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v uv     >/dev/null 2>&1 || { echo "[install] FAIL: uv not on PATH"; exit 1; }
command -v rustc  >/dev/null 2>&1 || { echo "[install] FAIL: rustc not on PATH after bootstrap"; exit 1; }
command -v protoc >/dev/null 2>&1 || { echo "[install] FAIL: protoc not on PATH after bootstrap"; exit 1; }
echo "[install] rustc:  $(rustc --version)"
echo "[install] protoc: $(protoc --version)"
echo "[install] $(date -Iseconds) starting"
echo "[install] nvcc: $(nvcc --version | tail -1)"
echo "[install] uv:   $(uv --version)"
echo "[install] CPATH=${CPATH:-<unset>}"

# ---- 0. libnuma1 (extract .deb to user dir, no sudo) ---------------------
LIBNUMA_DIR="$REPO_ROOT/.build/libnuma"
LIBNUMA_LIB_PATH="$LIBNUMA_DIR/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$LIBNUMA_LIB_PATH/libnuma.so.1" ]]; then
    echo "[install] [0/6] downloading + extracting libnuma1 to $LIBNUMA_DIR"
    mkdir -p "$LIBNUMA_DIR"
    (cd "$LIBNUMA_DIR" && apt-get download libnuma1 >/dev/null 2>&1 \
        && DEB=$(ls libnuma1*.deb) && dpkg-deb -x "$DEB" .)
    [[ -f "$LIBNUMA_LIB_PATH/libnuma.so.1" ]] || { echo "[install] FAIL: libnuma1 extract"; exit 1; }
fi
echo "[install] libnuma: $LIBNUMA_LIB_PATH/libnuma.so.1"

# ---- 1. Pin the SGLang ref ---------------------------------------------
git fetch origin main --quiet
HEAD_NOW="$(git rev-parse HEAD)"
if [[ "$HEAD_NOW" != "$SGLANG_PIN" ]]; then
    echo "[install] checking out SGLang $SGLANG_PIN onto local branch dsv4-deploy"
    git checkout -B dsv4-deploy "$SGLANG_PIN"
fi
echo "[install] SGLang ref: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# ---- 2. Initialize the uv project (root pyproject.toml + uv.lock) ------
if [[ ! -f pyproject.toml ]]; then
    echo "[install] uv init --bare (creates root pyproject.toml)"
    uv init --bare --name sglang-v4-flash-deploy --python "$PYTHON_VER" --no-readme --no-pin-python
fi

# ---- 3. SGLang (editable) ----------------------------------------------
# Pulls torch 2.11.0+cu130, sglang-kernel 0.4.2.post1, flashinfer_python 0.6.8.post1,
# tilelang 0.1.8, apache-tvm-ffi 0.1.9, sgl-deep-gemm 0.0.1, flash-attn-4, etc.
# Done BEFORE the from-source builds so the torch ABI is locked.
echo "[install] [3/6] uv add --editable ./python (sglang + full dep tree from main pyproject)"
uv add --editable ./python

# ---- 4. flashinfer-jit-cache (cu130 wheel, post1 to match flashinfer_python) ----
echo "[install] [4/6] uv add flashinfer-jit-cache==$FLASHINFER_JIT_CACHE_VER from $FLASHINFER_INDEX"
uv add --index "$FLASHINFER_INDEX" "flashinfer-jit-cache==$FLASHINFER_JIT_CACHE_VER"

# ---- 5. FlashMLA (local clone, no build isolation) ----------------------
FLASH_MLA_DIR="$REPO_ROOT/.build/flash-mla"
if [[ ! -d "$FLASH_MLA_DIR" ]]; then
    mkdir -p "$REPO_ROOT/.build"
    echo "[install] [5/6] cloning FlashMLA"
    git clone https://github.com/deepseek-ai/FlashMLA.git "$FLASH_MLA_DIR"
fi
git -C "$FLASH_MLA_DIR" submodule update --init --recursive --quiet
rm -rf "$FLASH_MLA_DIR/build"
echo "[install] [5/6] uv add --no-workspace --no-build-isolation flash-mla @ $FLASH_MLA_DIR (5-15 min)"
# --no-workspace because FlashMLA has setup.py only (no pyproject.toml); uv refuses
# to register a workspace member without one. As a path dep it builds via the
# legacy setuptools backend.
uv add --no-workspace --no-build-isolation --reinstall-package flash-mla "$FLASH_MLA_DIR"

# ---- 6. DeepGEMM (local clone, no build isolation) ----------------------
DEEPGEMM_DIR="$REPO_ROOT/.build/DeepGEMM"
if [[ ! -d "$DEEPGEMM_DIR" ]]; then
    echo "[install] [6/6] cloning DeepGEMM (branch $DEEPGEMM_BRANCH)"
    git clone https://github.com/sgl-project/DeepGEMM.git "$DEEPGEMM_DIR" -b "$DEEPGEMM_BRANCH"
fi
git -C "$DEEPGEMM_DIR" fetch --all --quiet
git -C "$DEEPGEMM_DIR" checkout "$DEEPGEMM_PIN" --quiet
git -C "$DEEPGEMM_DIR" submodule update --init --recursive --quiet
rm -rf "$DEEPGEMM_DIR/build" "$DEEPGEMM_DIR/dist" "$DEEPGEMM_DIR"/*.egg-info 2>/dev/null || true
echo "[install] [6/6] uv add --no-workspace --no-build-isolation deep-gemm @ $DEEPGEMM_DIR (10-20 min)"
uv add --no-workspace --no-build-isolation --reinstall-package deep-gemm "$DEEPGEMM_DIR"

# ---- Sanity import (with libnuma + conda libstdc++ on LD_LIBRARY_PATH) ---
# /opt/conda/lib must be first: the venv symlinks Python to /opt/conda/bin/python3,
# and conda packages (libicui18n.so.78 etc.) require CXXABI_1.3.15 from conda's
# libstdc++.so.6. The system /usr/lib/x86_64-linux-gnu/libstdc++.so.6 only has
# up to CXXABI_1.3.13. Without conda's lib first, sglang's transitive icu deps
# (transformers tokenizer chain) fail at import.
LD_LIBRARY_PATH="/opt/conda/lib:$LIBNUMA_LIB_PATH:${LD_LIBRARY_PATH:-}" .venv/bin/python - <<'PY'
import importlib, sys
mods = ['torch', 'sglang', 'flashinfer', 'deep_gemm', 'tilelang', 'tvm_ffi', 'flash_mla']
fails = []
for m in mods:
    try:
        x = importlib.import_module(m)
        print(f'[install]   OK  {m}={getattr(x, "__version__", "?")}')
    except Exception as e:
        print(f'[install]   FAIL {m}: {type(e).__name__}: {e}')
        fails.append(m)
try:
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    print(f'[install]   OK  sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM')
except Exception as e:
    print(f'[install]   FAIL DeepseekV4ForCausalLM: {type(e).__name__}: {e}')
    fails.append('DeepseekV4ForCausalLM')
import torch
print(f'[install]   torch.cuda.is_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'[install]   torch.version.cuda={torch.version.cuda}')
sys.exit(1 if fails else 0)
PY

echo "[install] $(date -Iseconds) done."
echo "[install] SGLang version: $(.venv/bin/python -c 'import sglang; print(sglang.__version__)')"
echo "[install] Next: bash serve_v4_flash.sh   (after vacating GPUs 0-3)"
