import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_explicit_stream(self) -> None:
    with sinks.TTYFixture() as tty:
        ap = autopage.AutoPager(tty.stream)
        stream = ap._paged_stream()
        self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=mock.ANY, universal_newlines=mock.ANY, encoding=mock.ANY, errors=mock.ANY, stdin=subprocess.PIPE, stdout=tty.stream)
        self.assertIs(stream, self.popen.return_value.stdin)