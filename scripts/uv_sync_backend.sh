#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/uv_sync_backend.sh [--backend auto|cuda|rocm] [--print-backend] [uv sync args...]

Select the correct Linux x86_64 PyTorch backend before running uv sync.

Examples:
  scripts/uv_sync_backend.sh
  scripts/uv_sync_backend.sh --backend cuda
  ACESTEP_TORCH_BACKEND=rocm scripts/uv_sync_backend.sh --offline
  scripts/uv_sync_backend.sh --print-backend
EOF
}

run_sync() {
    cd "$REPO_ROOT"
    uv sync "${sync_args[@]}"
}

install_rocm_packages() {
    local python_bin="$REPO_ROOT/.venv/bin/python"
    local -a pip_args=()

    if [[ ! -x "$python_bin" ]]; then
        echo "[Error] Expected uv environment at $python_bin after sync." >&2
        return 1
    fi

    for arg in "${sync_args[@]}"; do
        if [[ "$arg" == "--offline" ]]; then
            pip_args+=("--offline")
        fi
    done

    echo "[Setup] Reinstalling Linux x86_64 torch stack for ROCm 6.3"
    uv pip install --python "$python_bin" --force-reinstall "${pip_args[@]}" \
        --index-url https://download.pytorch.org/whl/rocm6.3 \
        torch==2.9.1+rocm6.3 \
        torchvision==0.24.1+rocm6.3 \
        torchaudio==2.9.1+rocm6.3 \
        pytorch-triton-rocm==3.5.1
}

has_visible_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

has_visible_rocm_gpu() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi -i >/dev/null 2>&1 && return 0
    fi
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo >/dev/null 2>&1 && return 0
    fi
    return 1
}

detect_backend() {
    local backend="${1:-auto}"

    if [[ "$backend" != "auto" ]]; then
        printf '%s\n' "$backend"
        return 0
    fi

    local has_nvidia=0
    local has_rocm=0
    has_visible_nvidia_gpu && has_nvidia=1
    has_visible_rocm_gpu && has_rocm=1

    if [[ "$has_nvidia" -eq 1 && "$has_rocm" -eq 0 ]]; then
        printf 'cuda\n'
        return 0
    fi
    if [[ "$has_rocm" -eq 1 && "$has_nvidia" -eq 0 ]]; then
        printf 'rocm\n'
        return 0
    fi
    if [[ "$has_rocm" -eq 1 && "$has_nvidia" -eq 1 ]]; then
        echo "[Error] Both NVIDIA and ROCm-capable runtimes were detected." >&2
        echo "[Error] Set ACESTEP_TORCH_BACKEND=cuda or ACESTEP_TORCH_BACKEND=rocm explicitly." >&2
        return 1
    fi

    echo "[Error] Could not detect an NVIDIA or ROCm GPU runtime on this machine." >&2
    echo "[Error] Set ACESTEP_TORCH_BACKEND=cuda or ACESTEP_TORCH_BACKEND=rocm explicitly." >&2
    return 1
}

selected_backend="${ACESTEP_TORCH_BACKEND:-auto}"
print_backend_only=0
declare -a sync_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            if [[ $# -lt 2 ]]; then
                echo "[Error] --backend requires a value" >&2
                usage >&2
                exit 1
            fi
            selected_backend="$2"
            shift 2
            ;;
        --print-backend)
            print_backend_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            sync_args+=("$1")
            shift
            ;;
    esac
done

case "$selected_backend" in
    auto|cuda|rocm) ;;
    *)
        echo "[Error] Unsupported backend: $selected_backend" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    if [[ "$selected_backend" != "auto" ]]; then
        echo "[Error] --backend is only supported on Linux x86_64." >&2
        exit 1
    fi
    if [[ "$print_backend_only" -eq 1 ]]; then
        printf 'native\n'
        exit 0
    fi
    run_sync
    exit 0
fi

resolved_backend="$(detect_backend "$selected_backend")"

if [[ "$print_backend_only" -eq 1 ]]; then
    printf '%s\n' "$resolved_backend"
    exit 0
fi

echo "[Setup] Linux x86_64 torch backend: $resolved_backend"
run_sync

if [[ "$resolved_backend" == "rocm" ]]; then
    install_rocm_packages
fi