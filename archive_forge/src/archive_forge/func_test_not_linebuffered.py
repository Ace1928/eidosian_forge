import unittest
import fixtures  # type: ignore
from typing import Any, Optional, Dict, List
import autopage
from autopage import command
def test_not_linebuffered(self) -> None:
    config = self._get_ap_config(line_buffering=False)
    self.assertTrue(config.color)
    self.assertFalse(config.line_buffering_requested)
    self.assertFalse(config.reset_terminal)