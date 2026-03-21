#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/uv_sync_backend.sh [options]

Create a uv-managed environment with the correct Python and PyTorch backend stack.

Options:
  --backend auto|cuda|rocm      GPU backend to install (default: auto)
  --python-version X.Y          Python version for uv-managed interpreter (default: 3.11)
  --venv-dir PATH               Virtual environment path (default: .venv)
  --offline                     Use offline mode for uv commands
  --print-backend               Print detected backend and exit
  -h, --help                    Show help

Examples:
  scripts/uv_sync_backend.sh
  scripts/uv_sync_backend.sh --backend cuda
  scripts/uv_sync_backend.sh --backend rocm --python-version 3.12
  ACESTEP_TORCH_BACKEND=rocm ACESTEP_PYTHON_VERSION=3.12 scripts/uv_sync_backend.sh --offline
  scripts/uv_sync_backend.sh --print-backend
EOF
}

ensure_uv_available() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "[Error] uv package manager not found in PATH." >&2
        echo "[Hint] Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
        return 1
    fi
}

install_cuda_stack() {
    local python_bin="$1"

    if [[ ! -x "$python_bin" ]]; then
        echo "[Error] Expected uv environment at $python_bin after sync." >&2
        return 1
    fi

    echo "[Setup] Installing CUDA 12.8 torch stack"
    uv pip install --python "$python_bin" "${uv_args[@]}" --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        torchaudio==2.10.0+cu128

    echo "[Setup] Installing CUDA dependencies from requirements.txt"
    uv pip install --python "$python_bin" "${uv_args[@]}" -r "$REPO_ROOT/requirements.txt"
}

install_rocm_packages() {
    local python_bin="$1"

    if [[ ! -x "$python_bin" ]]; then
        echo "[Error] Expected uv environment at $python_bin after setup." >&2
        return 1
    fi

    echo "[Setup] Reinstalling Linux x86_64 torch stack for ROCm 6.3"
    uv pip install --python "$python_bin" "${uv_args[@]}" --index-url https://download.pytorch.org/whl/rocm6.3 \
        torch==2.9.1+rocm6.3 \
        torchvision==0.24.1+rocm6.3 \
        torchaudio==2.9.1+rocm6.3 \
        pytorch-triton-rocm==3.5.1

    echo "[Setup] Installing ROCm dependencies from requirements-rocm-linux.txt"
    uv pip install --python "$python_bin" "${uv_args[@]}" -r "$REPO_ROOT/requirements-rocm-linux.txt"

    # requirements-rocm-linux.txt intentionally tracks ROCm essentials and may
    # lag a few project-level utility dependencies.
    uv pip install --python "$python_bin" "${uv_args[@]}" "diskcache" "typer-slim>=0.21.1" "lycoris-lora"
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
python_version="${ACESTEP_PYTHON_VERSION:-3.11}"
venv_dir=".venv"
print_backend_only=0
declare -a uv_args=()

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
        --python-version)
            if [[ $# -lt 2 ]]; then
                echo "[Error] --python-version requires a value" >&2
                usage >&2
                exit 1
            fi
            python_version="$2"
            shift 2
            ;;
        --venv-dir)
            if [[ $# -lt 2 ]]; then
                echo "[Error] --venv-dir requires a value" >&2
                usage >&2
                exit 1
            fi
            venv_dir="$2"
            shift 2
            ;;
        --offline)
            uv_args+=("--offline")
            shift
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
            echo "[Error] Unknown option: $1" >&2
            usage >&2
            exit 1
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

if [[ ! "$python_version" =~ ^3\.[0-9]+$ ]]; then
    echo "[Error] Unsupported Python version format: $python_version" >&2
    echo "[Hint] Use a major.minor value like 3.11 or 3.12." >&2
    exit 1
fi

ensure_uv_available

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    if [[ "$selected_backend" != "auto" && "$selected_backend" != "cuda" ]]; then
        echo "[Error] --backend rocm is only supported on Linux x86_64." >&2
        exit 1
    fi
    if [[ "$print_backend_only" -eq 1 ]]; then
        printf 'native\n'
        exit 0
    fi

    cd "$REPO_ROOT"
    echo "[Setup] Creating uv venv at $venv_dir with Python $python_version"
    uv python install "$python_version" "${uv_args[@]}"
    uv venv "$venv_dir" --python "$python_version" "${uv_args[@]}"
    uv sync "${uv_args[@]}"

    echo "[Setup] Environment ready: $venv_dir"
    exit 0
fi

resolved_backend="$(detect_backend "$selected_backend")"

if [[ "$print_backend_only" -eq 1 ]]; then
    printf '%s\n' "$resolved_backend"
    exit 0
fi

echo "[Setup] Linux x86_64 torch backend: $resolved_backend"
cd "$REPO_ROOT"

echo "[Setup] Ensuring Python $python_version is available via uv"
uv python install "$python_version" "${uv_args[@]}"

echo "[Setup] Creating venv at $venv_dir"
uv venv "$venv_dir" --python "$python_version" "${uv_args[@]}"

python_bin="$REPO_ROOT/$venv_dir/bin/python"

if [[ "$resolved_backend" == "rocm" ]]; then
    install_rocm_packages "$python_bin"
else
    install_cuda_stack "$python_bin"
fi

echo "[Setup] Installing ACE-Step package into selected environment"
uv pip install --python "$python_bin" "${uv_args[@]}" --editable "$REPO_ROOT" --no-deps

echo "[Setup] Verifying torch backend"
if [[ "$resolved_backend" == "rocm" ]]; then
    uv run --python "$python_bin" --no-sync python -c "import torch; assert torch.version.hip is not None, 'Expected ROCm torch build'; print({'torch': torch.__version__, 'hip': torch.version.hip})"
else
    uv run --python "$python_bin" --no-sync python -c "import torch; assert torch.version.cuda is not None, 'Expected CUDA torch build'; print({'torch': torch.__version__, 'cuda': torch.version.cuda})"
fi

if [[ "$venv_dir" == ".venv" ]]; then
    echo "[Setup] Done. You can run: uv run acestep"
else
    echo "[Setup] Done. Use this interpreter for ACE-Step: $python_bin"
fi

