import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_list_int(self) -> None:
    self.assertRaises(TypeError, command.get_pager_command, ['FOO', 42])