import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_less_env_reset(self) -> None:
    config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=True)
    less_env = self._env(config)['LESS']
    self.assertEqual('R', less_env)