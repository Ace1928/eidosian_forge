import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
def test_addstr(self):
    self.repl.complete = mock.Mock(True)
    self.repl.s = 'foo'
    self.repl.addstr('bar')
    self.assertEqual(self.repl.s, 'foobar')
    self.repl.cpos = 3
    self.repl.addstr('buzz')
    self.assertEqual(self.repl.s, 'foobuzzbar')