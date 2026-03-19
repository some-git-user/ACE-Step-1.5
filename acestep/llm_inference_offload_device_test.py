"""Unit tests for temporary PT LM accelerator routing in generate_from_formatted_prompt."""

import unittest
from unittest.mock import MagicMock, patch

try:
    from acestep.llm_inference import LLMHandler
    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as exc:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(exc)
    LLMHandler = None  # type: ignore[assignment]


@unittest.skipIf(not _IMPORT_OK, f"llm_inference import failed: {_IMPORT_ERROR}")
class TestGenerateFromFormattedPromptOffloadDeviceRouting(unittest.TestCase):
    """Verify temporary device routing in PT offload mode."""

    def test_routes_to_cuda_temporarily_and_restores_device(self):
        handler = LLMHandler()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.offload_to_cpu = True
        handler.device = "cpu"
        handler.llm = MagicMock()
        handler.llm_tokenizer = MagicMock()

        observed_devices: list[str] = []

        def _fake_run_pt(**_kwargs):
            observed_devices.append(handler.device)
            return "ok"

        handler._run_pt = MagicMock(side_effect=_fake_run_pt)
        handler._clear_accelerator_cache = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            text, status = handler.generate_from_formatted_prompt("prompt", cfg={"temperature": 0.6})

        self.assertEqual(text, "ok")
        self.assertIn("Generated successfully", status)
        self.assertEqual(observed_devices, ["cuda"])
        self.assertEqual(handler.device, "cpu")

    def test_keeps_cpu_when_no_accelerator_is_available(self):
        handler = LLMHandler()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.offload_to_cpu = True
        handler.device = "cpu"
        handler.llm = MagicMock()
        handler.llm_tokenizer = MagicMock()

        observed_devices: list[str] = []

        def _fake_run_pt(**_kwargs):
            observed_devices.append(handler.device)
            return "ok"

        handler._run_pt = MagicMock(side_effect=_fake_run_pt)
        handler._clear_accelerator_cache = MagicMock()

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.xpu.is_available", return_value=False, create=True), \
             patch("torch.backends.mps.is_available", return_value=False, create=True):
            text, status = handler.generate_from_formatted_prompt("prompt", cfg={"temperature": 0.6})

        self.assertEqual(text, "ok")
        self.assertIn("Generated successfully", status)
        self.assertEqual(observed_devices, ["cpu"])
        self.assertEqual(handler.device, "cpu")


if __name__ == "__main__":
    unittest.main()
