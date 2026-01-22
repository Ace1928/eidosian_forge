import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_line_buffering(self) -> None:
    ap = autopage.AutoPager(line_buffering=True)
    stream = ap._paged_stream()
    self.popen.assert_called_once_with(mock.ANY, env=mock.ANY, bufsize=1, universal_newlines=True, encoding=mock.ANY, errors=mock.ANY, stdin=subprocess.PIPE, stdout=None)
    self.assertIs(stream, self.popen.return_value.stdin)