"""Unit tests for LLM GPU offload helpers in inference.py.

Tests cover ``_lm_gpu_context_enter`` and ``_lm_gpu_context_exit``:
- Normal move from CPU to GPU and back.
- Skip when LLM is already on GPU.
- Skip when backend is not "pt" (vllm/mlx).
- Skip when LLM is None.
- Graceful handling of OOM on move.
- Exit is idempotent (safe when LLM is already on CPU).
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Optional

import torch
import torch.nn as nn

try:
    from acestep.inference import (
        _lm_gpu_context_enter,
        _lm_gpu_context_exit,
        format_sample,
        understand_music,
    )
    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as exc:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(exc)
    _lm_gpu_context_enter = None  # type: ignore[assignment]
    _lm_gpu_context_exit = None   # type: ignore[assignment]
    format_sample = None  # type: ignore[assignment]
    understand_music = None  # type: ignore[assignment]


def _make_llm_handler(
    backend: str = "pt",
    device: str = "cpu",
    param_device: str = "cpu",
    llm: Optional[nn.Module] = None,
) -> MagicMock:
    """Return a minimal LLMHandler-like mock."""
    handler = MagicMock()
    handler.llm_backend = backend
    handler.device = device
    handler.offload_to_cpu = True
    if llm is not None:
        handler.llm = llm
    else:
        # Fake linear layer with a single cpu parameter.
        param = MagicMock(spec=torch.nn.Parameter)
        dev = MagicMock()
        dev.type = param_device
        param.device = dev
        fake_llm = MagicMock(spec=nn.Module)
        fake_llm.parameters.side_effect = lambda: iter([param])
        handler.llm = fake_llm
    return handler


def _make_mutable_device_llm(initial_device: str = "cpu") -> tuple[MagicMock, MagicMock]:
    """Build a fake nn.Module whose ``to(device)`` updates parameter device type."""
    fake_llm = MagicMock(spec=nn.Module)
    param = MagicMock(spec=torch.nn.Parameter)
    param.device.type = initial_device
    fake_llm.parameters.side_effect = lambda: iter([param])

    def _to(device):
        param.device.type = str(device).split(":")[0]
        return fake_llm

    fake_llm.to.side_effect = _to
    return fake_llm, param


@unittest.skipIf(not _IMPORT_OK, f"inference import failed: {_IMPORT_ERROR}")
class TestLmGpuContextEnter(unittest.TestCase):
    """Tests for ``_lm_gpu_context_enter``."""

    def test_returns_false_when_llm_handler_is_none(self):
        result = _lm_gpu_context_enter(None, "cuda")
        self.assertFalse(result)

    def test_returns_false_when_llm_is_none(self):
        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = None
        result = _lm_gpu_context_enter(handler, "cuda")
        self.assertFalse(result)

    def test_returns_false_for_vllm_backend(self):
        handler = _make_llm_handler(backend="vllm")
        result = _lm_gpu_context_enter(handler, "cuda")
        self.assertFalse(result)

    def test_returns_false_for_mlx_backend(self):
        handler = _make_llm_handler(backend="mlx")
        result = _lm_gpu_context_enter(handler, "cuda")
        self.assertFalse(result)

    def test_returns_false_when_already_on_gpu(self):
        handler = _make_llm_handler(param_device="cuda")
        result = _lm_gpu_context_enter(handler, "cuda")
        self.assertFalse(result)

    def test_moves_to_gpu_when_on_cpu_and_returns_true(self):
        fake_llm = MagicMock(spec=nn.Module)
        param = MagicMock(spec=torch.nn.Parameter)
        param.device.type = "cpu"
        fake_llm.parameters.return_value = iter([param])
        moved_llm = MagicMock()
        fake_llm.to.return_value = moved_llm

        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = fake_llm

        result = _lm_gpu_context_enter(handler, "cuda")

        self.assertTrue(result)
        fake_llm.to.assert_called_once_with("cuda")
        self.assertIs(handler.llm, moved_llm)
        self.assertEqual(handler.device, "cuda")

    def test_returns_false_when_move_raises_oom(self):
        fake_llm = MagicMock(spec=nn.Module)
        param = MagicMock(spec=torch.nn.Parameter)
        param.device.type = "cpu"
        fake_llm.parameters.return_value = iter([param])
        fake_llm.to.side_effect = torch.cuda.OutOfMemoryError("OOM")

        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = fake_llm

        result = _lm_gpu_context_enter(handler, "cuda")

        self.assertFalse(result)

    def test_returns_false_when_model_has_no_parameters(self):
        fake_llm = MagicMock(spec=nn.Module)
        fake_llm.parameters.return_value = iter([])  # empty

        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = fake_llm

        result = _lm_gpu_context_enter(handler, "cuda")
        self.assertFalse(result)


@unittest.skipIf(not _IMPORT_OK, f"inference import failed: {_IMPORT_ERROR}")
class TestLmGpuContextExit(unittest.TestCase):
    """Tests for ``_lm_gpu_context_exit``."""

    def test_noop_when_handler_is_none(self):
        # Should not raise.
        _lm_gpu_context_exit(None)

    def test_noop_when_llm_is_none(self):
        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = None
        _lm_gpu_context_exit(handler)  # Should not raise.

    def test_noop_for_vllm_backend(self):
        handler = _make_llm_handler(backend="vllm", param_device="cuda")
        _lm_gpu_context_exit(handler)  # Should not raise.

    def test_noop_when_already_on_cpu(self):
        handler = _make_llm_handler(param_device="cpu")
        _lm_gpu_context_exit(handler)
        # .to() must not have been called since it was already on CPU.
        handler.llm.to.assert_not_called()

    def test_moves_to_cpu_when_on_gpu(self):
        fake_llm = MagicMock(spec=nn.Module)
        param = MagicMock(spec=torch.nn.Parameter)
        param.device.type = "cuda"
        fake_llm.parameters.return_value = iter([param])
        restored_llm = MagicMock()
        fake_llm.to.return_value = restored_llm

        handler = MagicMock()
        handler.llm_backend = "pt"
        handler.llm = fake_llm

        with patch("torch.cuda.is_available", return_value=False):
            _lm_gpu_context_exit(handler)

        fake_llm.to.assert_called_once_with("cpu")
        self.assertIs(handler.llm, restored_llm)
        self.assertEqual(handler.device, "cpu")


@unittest.skipIf(not _IMPORT_OK, f"inference import failed: {_IMPORT_ERROR}")
class TestFormatSampleOffloadContext(unittest.TestCase):
    """Regression tests for ``format_sample`` accelerator usage in offload mode."""

    def test_format_sample_moves_llm_to_cuda_then_back_when_offload_enabled(self):
        fake_llm, _ = _make_mutable_device_llm(initial_device="cpu")
        handler = MagicMock()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.offload_to_cpu = True
        handler.device = "cpu"
        handler.llm = fake_llm
        handler.format_sample_from_input.return_value = (
            {
                "caption": "enhanced caption",
                "lyrics": "enhanced lyrics",
                "bpm": 120,
                "duration": 30,
                "keyscale": "C Major",
                "language": "en",
                "timesignature": "4",
            },
            "ok",
        )

        with patch("torch.cuda.is_available", return_value=True):
            result = format_sample(handler, "c", "l")

        self.assertTrue(result.success)
        self.assertEqual(fake_llm.to.call_count, 2)
        fake_llm.to.assert_any_call("cuda")
        fake_llm.to.assert_any_call("cpu")
        self.assertEqual(handler.device, "cpu")

    def test_format_sample_does_not_move_llm_when_offload_disabled(self):
        fake_llm, _ = _make_mutable_device_llm(initial_device="cpu")
        handler = MagicMock()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.offload_to_cpu = False
        handler.device = "cpu"
        handler.llm = fake_llm
        handler.format_sample_from_input.return_value = (
            {"caption": "enhanced", "lyrics": "lyrics"},
            "ok",
        )

        with patch("torch.cuda.is_available", return_value=True):
            result = format_sample(handler, "c", "l")

        self.assertTrue(result.success)
        fake_llm.to.assert_not_called()


@unittest.skipIf(not _IMPORT_OK, f"inference import failed: {_IMPORT_ERROR}")
class TestUnderstandMusicOffloadContext(unittest.TestCase):
    """Regression tests for ``understand_music`` accelerator usage in offload mode."""

    def test_understand_music_moves_llm_to_cuda_then_back_when_offload_enabled(self):
        fake_llm, _ = _make_mutable_device_llm(initial_device="cpu")
        handler = MagicMock()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.offload_to_cpu = True
        handler.device = "cpu"
        handler.llm = fake_llm
        handler.understand_audio_from_codes.return_value = (
            {
                "caption": "analyzed caption",
                "lyrics": "analyzed lyrics",
                "bpm": 120,
                "duration": 30,
                "keyscale": "C Major",
                "language": "en",
                "timesignature": "4/4",
            },
            "ok",
        )

        with patch("torch.cuda.is_available", return_value=True):
            result = understand_music(handler, "<|audio_code_1|>")

        self.assertTrue(result.success)
        self.assertEqual(fake_llm.to.call_count, 2)
        fake_llm.to.assert_any_call("cuda")
        fake_llm.to.assert_any_call("cpu")
        self.assertEqual(handler.device, "cpu")


if __name__ == "__main__":
    unittest.main()
