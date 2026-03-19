"""Unit tests for LM startup selection helpers."""

import os
import unittest
from unittest.mock import patch

from acestep.pipeline_lm_selection import env_flag, maybe_downgrade_lm_model


class TestEnvFlag(unittest.TestCase):
    """Verify boolean environment parsing for startup overrides."""

    def test_returns_default_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_flag("FLAG", False))
            self.assertTrue(env_flag("FLAG", True))

    def test_parses_truthy_values(self):
        for value in ["1", "true", "yes", "y", "on"]:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"FLAG": value}, clear=True):
                    self.assertTrue(env_flag("FLAG", False))

    def test_parses_falsy_values(self):
        for value in ["0", "false", "no", "n", "off"]:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"FLAG": value}, clear=True):
                    self.assertFalse(env_flag("FLAG", True))

    def test_invalid_value_falls_back_to_default(self):
        with patch.dict(os.environ, {"FLAG": "maybe"}, clear=True):
            self.assertTrue(env_flag("FLAG", True))
            self.assertFalse(env_flag("FLAG", False))


class TestMaybeDowngradeLmModel(unittest.TestCase):
    """Verify 4B LM safety downgrade policy."""

    def test_leaves_non_4b_model_unchanged(self):
        self.assertEqual(
            maybe_downgrade_lm_model("acestep-5Hz-lm-1.7B", 16.0, 20.0),
            "acestep-5Hz-lm-1.7B",
        )

    def test_downgrades_4b_below_threshold(self):
        self.assertEqual(
            maybe_downgrade_lm_model("acestep-5Hz-lm-4B", 16.0, 20.0),
            "acestep-5Hz-lm-1.7B",
        )

    def test_keeps_4b_at_or_above_threshold(self):
        self.assertEqual(
            maybe_downgrade_lm_model("acestep-5Hz-lm-4B", 24.0, 20.0),
            "acestep-5Hz-lm-4B",
        )

    def test_override_keeps_4b_below_threshold(self):
        self.assertEqual(
            maybe_downgrade_lm_model(
                "acestep-5Hz-lm-4B",
                16.0,
                20.0,
                allow_unsupported_lm=True,
            ),
            "acestep-5Hz-lm-4B",
        )


if __name__ == "__main__":
    unittest.main()
