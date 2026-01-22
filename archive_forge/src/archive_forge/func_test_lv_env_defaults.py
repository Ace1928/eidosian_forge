import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_lv_env_defaults(self) -> None:
    config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
    lv_env = self._env(config)['LV']
    self.assertEqual('-c', lv_env)