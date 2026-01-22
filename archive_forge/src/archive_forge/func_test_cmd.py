import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_cmd(self) -> None:
    self.assertEqual(['less', '-r', '+F'], self.cmd.command())