"""LM startup selection helpers for the Gradio pipeline.

This module keeps LM-size safety decisions separate from the main startup
pipeline so the policy can be tested without importing the full UI stack.
"""

import os
from typing import Optional


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean flag from the environment.

    Truthy values: ``1``, ``true``, ``yes``, ``y``, ``on``.
    Falsy values: ``0``, ``false``, ``no``, ``n``, ``off``.
    Any other value falls back to ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def maybe_downgrade_lm_model(
    lm_model_path: Optional[str],
    gpu_memory_gb: float,
    vram_auto_offload_threshold_gb: float,
    *,
    allow_unsupported_lm: bool = False,
) -> Optional[str]:
    """Return the effective LM model path after applying safety downgrade rules.

    On GPUs below the auto-offload threshold, selecting the 4B LM can leave too
    little memory headroom for generation. By default, the startup flow falls
    back to the 1.7B sibling. Users can explicitly opt out of this downgrade by
    setting ``allow_unsupported_lm=True``.
    """
    if allow_unsupported_lm:
        return lm_model_path
    if not lm_model_path or "4B" not in lm_model_path:
        return lm_model_path
    if not (0 < gpu_memory_gb < vram_auto_offload_threshold_gb):
        return lm_model_path
    return lm_model_path.replace("4B", "1.7B")
