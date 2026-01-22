import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_lv_env_linebuffered_nocolor(self) -> None:
    config = command.PagerConfig(color=False, line_buffering_requested=True, reset_terminal=False)
    self.assertNotIn('LV', self._env(config))