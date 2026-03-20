"""Unit tests for generation interface default resolution."""

import os
import unittest
from unittest.mock import patch

from acestep.ui.gradio.interfaces.generation_defaults import compute_init_defaults


class _StubGPUConfig:
    """Minimal GPU config stub required by ``compute_init_defaults`` tests."""

    max_duration_with_lm = 240
    max_duration_without_lm = 180
    max_batch_size_with_lm = 4
    max_batch_size_without_lm = 2
    init_lm_default = True
    offload_to_cpu_default = False
    offload_dit_to_cpu_default = False
    quantization_default = "none"
    compile_model_default = False
    lm_backend_restriction = "none"
    recommended_backend = "pt"
    recommended_lm_model = "acestep-5Hz-lm-1.7B"


class GenerationDefaultsEnvTests(unittest.TestCase):
    """Verify generation defaults sourced from environment variables."""

    def test_uses_env_value_for_lm_negative_prompt_default(self) -> None:
        """`ACESTEP_LM_NEGATIVE_PROMPT` should populate UI default when set."""

        init_params = {"gpu_config": _StubGPUConfig()}
        with patch.dict(os.environ, {"ACESTEP_LM_NEGATIVE_PROMPT": "no hiss, no clipping"}, clear=False):
            defaults = compute_init_defaults(init_params=init_params, language="en")

        self.assertEqual("no hiss, no clipping", defaults["lm_negative_prompt_default"])

    def test_falls_back_to_no_user_input_when_env_is_blank(self) -> None:
        """Blank env values should resolve to the stable default text."""

        init_params = {"gpu_config": _StubGPUConfig()}
        with patch.dict(os.environ, {"ACESTEP_LM_NEGATIVE_PROMPT": "   "}, clear=False):
            defaults = compute_init_defaults(init_params=init_params, language="en")

        self.assertEqual("NO USER INPUT", defaults["lm_negative_prompt_default"])

    def test_uses_env_value_for_lm_codes_strength_default(self) -> None:
        """`ACESTEP_LM_CODES_STRENGTH` should populate slider default when valid."""

        init_params = {"gpu_config": _StubGPUConfig()}
        with patch.dict(os.environ, {"ACESTEP_LM_CODES_STRENGTH": "0.73"}, clear=False):
            defaults = compute_init_defaults(init_params=init_params, language="en")

        self.assertEqual(0.73, defaults["lm_codes_strength_default"])

    def test_clamps_lm_codes_strength_default_to_slider_range(self) -> None:
        """`ACESTEP_LM_CODES_STRENGTH` should be clamped into 0.0..1.0 range."""

        init_params = {"gpu_config": _StubGPUConfig()}
        with patch.dict(os.environ, {"ACESTEP_LM_CODES_STRENGTH": "2.5"}, clear=False):
            defaults_high = compute_init_defaults(init_params=init_params, language="en")
        with patch.dict(os.environ, {"ACESTEP_LM_CODES_STRENGTH": "-0.2"}, clear=False):
            defaults_low = compute_init_defaults(init_params=init_params, language="en")

        self.assertEqual(1.0, defaults_high["lm_codes_strength_default"])
        self.assertEqual(0.0, defaults_low["lm_codes_strength_default"])

    def test_falls_back_when_lm_codes_strength_is_invalid(self) -> None:
        """Invalid `ACESTEP_LM_CODES_STRENGTH` should use stable default value."""

        init_params = {"gpu_config": _StubGPUConfig()}
        with patch.dict(os.environ, {"ACESTEP_LM_CODES_STRENGTH": "not-a-number"}, clear=False):
            defaults = compute_init_defaults(init_params=init_params, language="en")

        self.assertEqual(1.0, defaults["lm_codes_strength_default"])


if __name__ == "__main__":
    unittest.main()
