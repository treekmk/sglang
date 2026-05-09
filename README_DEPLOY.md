# DeepSeek-V4-Flash on SGLang (B200, TP=4) — local deploy

Serves `deepseek-ai/DeepSeek-V4-Flash` from this repo's source tree at `http://localhost:30000/v1`,
on GPUs 0–3 (4× B200 ~192 GB each, sm_100). Exposes `/v1/score` with the `label_token_ids`
extension that zbench needs for lossless per-digit logprobs.

## Pinned versions

| Component | Version | Notes |
|---|---|---|
| SGLang ref | `47e9ec11ad7a12b85aff4ed6efc5eb3a43961d16` (`origin/main`) | Post-PR #23882 (commit `35870d55a` "Deepseek V4"); V4 Flash + Pro support landed on main 2026-05-07. Local branch named `dsv4-deploy`. |
| Python | 3.11 | uv venv at `./.venv` |
| `torch` | 2.11.0+cu130 | Pinned by `python/pyproject.toml`. cu130 wheel matches host nvcc cu13.0 — torch's strict major-version guard is satisfied without any monkey-patch. |
| `sglang-kernel` | 0.4.2.post1 | Pinned by main's pyproject (renamed from `sgl-kernel` on V4 branch) |
| `flashinfer_python` | 0.6.8.post1 | Pinned by main's pyproject |
| `flashinfer-jit-cache` | 0.6.8.post1 (cu130 wheel) | Added separately via cu130 index — matches host CUDA at runtime, no JIT recompile needed |
| `tilelang` | 0.1.8 | Pinned by main's pyproject |
| `apache-tvm-ffi` | 0.1.9 | Pinned by main's pyproject (no manual re-pin step required) |
| `sgl-deep-gemm` | 0.0.1 | SGLang-bundled prebuilt stub; the real `deep_gemm` Python module is provided by the from-source build below |
| FlashMLA | `deepseek-ai/FlashMLA` HEAD | from-source build against cu130 + torch 2.11 |
| DeepGEMM | `sgl-project/DeepGEMM` @ `7f2a70` (branch `release-0426`) | from-source build against cu130 |

The recipe mirrors `docker/deepseek_v4_b200.Dockerfile` on `origin/deepseek_v4`, with these
deviations: pin **main** (not `origin/deepseek_v4`); host CUDA toolkit is 13.0 (Dockerfile assumes
12.9); `flashinfer-jit-cache` cu130 wheel (Dockerfile uses cu129); `DeepGEMM` cloned from
branch `release-0426` (Dockerfile says `release` which no longer exists).

## Install

```bash
# pre-reqs already satisfied on this host: nvcc at /usr/local/cuda-13.0/bin/nvcc, uv at ~/.local/bin/uv
bash install_v4_flash.sh   # 20–40 min on a clean host (mostly FlashMLA + DeepGEMM compiles)
```

The installer uses **`uv add`** end-to-end — every dep lands in a top-level `pyproject.toml` + `uv.lock`
generated at the repo root. That gives a single declarative manifest (and a reproducible lock) for the
deploy, separate from `python/pyproject.toml` (the SGLang package itself) and `sgl-kernel/pyproject.toml`.

Re-running the script is idempotent: `uv add` is a no-op when the pin matches; `--reinstall-package`
forces a fresh build for the from-source packages. The git submodule + clone steps reuse `.build/`.

What it does, in order:

1. Extract `libnuma1.deb` to `.build/libnuma/` (no sudo) — needed by `sgl_kernel/sm100/common_ops.abi3.so`.
2. `git fetch origin main && git checkout -B dsv4-deploy 47e9ec11a…` — pin the SGLang ref.
3. `uv init --bare --name sglang-v4-flash-deploy --python 3.11 --no-readme --no-pin-python`.
4. `uv add --editable ./python` — installs SGLang plus the full dep tree (torch 2.11.0+cu130, sglang-kernel 0.4.2.post1, tilelang 0.1.8, apache-tvm-ffi 0.1.9, sgl-deep-gemm 0.0.1, flashinfer_python 0.6.8.post1, etc.). Done first so torch is at the right version before any from-source build.
5. `uv add --index https://flashinfer.ai/whl/cu130 flashinfer-jit-cache==0.6.8.post1` — cu130 JIT cache.
6. Clone `deepseek-ai/FlashMLA` to `.build/flash-mla`; wipe stale `build/`; `uv add --no-workspace --no-build-isolation --reinstall-package flash-mla "$REPO_ROOT/.build/flash-mla"` — builds against cu130 + torch 2.11.
7. Clone `sgl-project/DeepGEMM` to `.build/DeepGEMM` from branch `release-0426`, checkout `7f2a70`; `uv add --no-workspace --no-build-isolation --reinstall-package deep-gemm "$REPO_ROOT/.build/DeepGEMM"`.

## Launch

```bash
bash serve_v4_flash.sh
```

Behavior:

- Sets `CUDA_VISIBLE_DEVICES=0,1,2,3`, `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1`, `CUDA_HOME=/usr/local/cuda-13.0`, and prepends `.build/libnuma/usr/lib/x86_64-linux-gnu/` to `LD_LIBRARY_PATH`.
- Refuses to start if port 30000 is already taken or if a previous launch's pid is still alive.
- Warns (does not block) if GPUs 0–3 collectively hold > 30 GB — likely means the vLLM on port 10000 is still resident; SGLang will OOM on weight load.
- Backgrounds `python -m sglang.launch_server …`, writes its pid to `serve_v4_flash.pid` and stdout/stderr to `serve_v4_flash.log`.
- Polls `GET /v1/models` until 200 OK or the deadline (default 3600 s, override via `READY_DEADLINE_SEC=`).
- Prints recent log lines once per minute while waiting.

Launch flags (B200 Low-Latency recipe from `test/manual/dsv4/test_b200_flash.py:TestB200FlashLowLatency`,
identical between `origin/main` and `origin/deepseek_v4`, with `--mem-fraction-static 0.85` from the deploy spec):

```
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tp 4 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune
```

Override knobs (env vars): `HOST`, `PORT`, `MODEL`, `MEM_FRACTION_STATIC`, `CUDA_VISIBLE_DEVICES`, `READY_DEADLINE_SEC`.

## Smoke test

```bash
source .venv/bin/activate    # optional; the script uses only stdlib
python smoke_v4_flash.py
```

Hits `/v1/models`, `/v1/completions`, and `/v1/score` (with the deploy spec's label-token payload).
Prints `PASS` and exits 0 on success.

## Stop cleanly

```bash
kill -TERM $(cat serve_v4_flash.pid)
```

`serve_v4_flash.sh` runs the server under `setsid` so SIGTERM reaches the whole process tree (the
launcher + TP workers).

## Known issues

- **No NVIDIA B200 CI for V4.** All V4 tests in `test/manual/dsv4/` are manual; the only nightly V4
  CI is AMD ROCm. We're trusting the upstream V4 PR (#23882) + the V4 branch's release Dockerfile,
  not green CI.
- **CUDA 13 moved libcudacxx headers, CUTLASS guards on an nvcc-only macro.** CUDA 13 relocated
  `<cuda/std/...>` to `<cccl/cuda/std/...>` (provided by `cuda-cccl-13-0`, installed at
  `/usr/local/cuda-13.0/include/cccl/`). CUTLASS in FlashMLA's submodule (v4.3.0) does switch on
  `__CUDACC_VER_MAJOR__ >= 13`, but that macro is only defined while *nvcc* is compiling. Host-side
  `.cpp` files (compiled by `c++`) fall through to the cu12 path and fail with `fatal error:
  cuda/std/utility: No such file or directory`. The installer exports
  `CPATH=$CUDA_HOME/include/cccl` so both gcc and nvcc resolve `<cuda/std/...>` to the cu13 location.
- **`libnuma.so.1` missing on host.** `sgl_kernel/sm100/common_ops.abi3.so` is dynamically linked
  against `libnuma.so.1` but the host doesn't ship it. Without it, ALL model loads fail at sglang
  import time (`ImportError: libnuma.so.1: cannot open shared object file`). The installer downloads
  `libnuma1_2.0.14-3ubuntu2_amd64.deb` via `apt-get download` (no sudo) and extracts to
  `.build/libnuma/`. `serve_v4_flash.sh` adds `.build/libnuma/usr/lib/x86_64-linux-gnu/` to
  `LD_LIBRARY_PATH` so sgl_kernel can dlopen it. Set this manually if you launch sglang outside the
  serve script.
- **System `libstdc++.so.6` is too old for conda-shipped icu.** uv's venv symlinks Python to
  `/opt/conda/bin/python3` (the system Python 3.11). Conda packages on the import path
  (e.g. `libicui18n.so.78` reached via the transformers tokenizer chain) need `CXXABI_1.3.15`. The
  system `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (gcc-11 era) only has up to `CXXABI_1.3.13`;
  `/opt/conda/lib/libstdc++.so.6` has `1.3.15`. Symptom: `ImportError: …libstdc++.so.6: version
  'CXXABI_1.3.15' not found (required by /opt/conda/lib/python3.11/lib-dynload/../.././libicui18n.so.78)`
  at sglang import. `serve_v4_flash.sh` prepends `/opt/conda/lib` to `LD_LIBRARY_PATH` to force
  the loader to pick conda's libstdc++ first. If you launch python directly, do the same.
- **`sglang-grpc` Rust crate needs `protoc` at build time.** Main's `python/` no longer ships a
  Python `setup.py` for proto compilation; the Rust router crate `rust/sglang-grpc/build.rs` calls
  `tonic_build::compile_protos` which invokes `protoc` via `prost-build`. Without it: `Could not
  find protoc` and `cargo rustc … failed with code 101`. The installer downloads
  `protoc-29.3-linux-x86_64.zip` to `.build/protoc/` (no sudo) and prepends `bin/` to PATH.
- **Rust toolchain needed for sglang's editable install.** `uv add --editable ./python` triggers a
  `cargo build` of `rust/sglang-grpc` (pyo3 cdylib). The installer bootstraps rustup non-interactively
  to `~/.cargo` and `~/.rustup` (no sudo, no shell-rc modification) and prepends `~/.cargo/bin` to
  PATH. Build deps (in `python/pyproject.toml`'s `build-system.requires`) are `setuptools-rust>=1.10`.
  Note: V4 branch (`origin/deepseek_v4` @ `1b497c7a0`) had a custom Python `setup.py` for proto
  compilation and didn't need the Rust path; main consolidated proto handling into the Rust crate.
- **`uv add` workspace auto-adoption + missing pyproject.toml.** `uv add <local-path>` registers the
  path as a workspace member by default. FlashMLA and DeepGEMM ship only `setup.py` (no
  `pyproject.toml`), and uv refuses to take a workspace member without one. Pass `--no-workspace`
  on those calls — the installer does this. The `python/` workspace member is fine because SGLang
  itself ships `python/pyproject.toml`.
- **DeepGEMM `release` branch was renamed.** The Dockerfile says `git clone -b release …` but the
  upstream `sgl-project/DeepGEMM` no longer has a `release` branch. Commit `7f2a70` lives on
  `release-0426` and `dev-0426`; the install script clones from `release-0426`.
- **Install order: sglang BEFORE FlashMLA (deviates from Dockerfile).** The B200 Dockerfile installs
  FlashMLA first, then `pip install -e sglang/python/`. That works there because the base image
  (`lmsysorg/sglang:v0.5.7`) already has the correct torch resident — FlashMLA links against it. In
  a fresh venv, the order matters: if FlashMLA runs first, `--no-build-isolation` lets it pull a
  transient newer torch as a setup-time dep; sglang's pyproject (`torch==2.11.0`) then downgrades
  torch, leaving FlashMLA's `.so` referencing torch symbols that don't exist in the resident torch.
  Symptom: `ImportError: undefined symbol: _ZN3c104cuda…` at server startup. Fix: install sglang
  first; the script does this.
- **`DeepseekV4ForCausalLMNextN` lives in a separate file.** It's in
  `python/sglang/srt/models/deepseek_v4_nextn.py`, not in `deepseek_v4.py`. If you write a
  one-liner sanity check, import each from its own module.
- **uv project layout.** `install_v4_flash.sh` creates `pyproject.toml` and `uv.lock` at the **repo
  root**, separate from `python/pyproject.toml` (SGLang itself) and `sgl-kernel/pyproject.toml`.
  They are untracked by default so `git checkout main` won't touch them; commit them onto
  `dsv4-deploy` (or any branch) if you want the deploy manifest pinned in version control. To start
  over: `rm -rf .venv pyproject.toml uv.lock` then re-run the installer.

## Dropped from earlier deploy attempts (no longer applicable)

The earlier pin to `origin/deepseek_v4` @ `1b497c7a0` (which used `torch==2.9.1+cu128`) required two
extra workarounds that are no longer needed on main:

- **`sitecustomize.py` torch CUDA-version monkey-patch** — needed when torch's `cu128` ABI conflicted
  with the host's `cu13` nvcc, because torch's `_check_cuda_version` raises hard on a major-version
  mismatch and there's no env-var bypass. Main pins `torch==2.11.0+cu130`, which matches `cu13.0`,
  so the check passes naturally.
- **`apache-tvm-ffi==0.1.9` re-pin step** — the V4 branch didn't pin tvm-ffi and DeepGEMM's
  `install.sh` would bump it to 0.1.10, breaking tilelang 0.1.8 ABI. Main's pyproject pins 0.1.9
  directly, so no re-pin is needed.

## Branch / git hygiene

`install_v4_flash.sh` checks out a local branch `dsv4-deploy` at the pinned SHA on origin/main. The
deploy files (`install_v4_flash.sh`, `serve_v4_flash.sh`, `smoke_v4_flash.py`, `README_DEPLOY.md`,
`pyproject.toml`, `uv.lock`) live at the repo root and are deliberately not committed yet — review
and `git add` them onto whatever branch makes sense for your workflow. To return to `main`:

```bash
git checkout main          # untracked deploy files survive the switch
```

To remove the venv and built artifacts:

```bash
rm -rf .venv .build pyproject.toml uv.lock
```
