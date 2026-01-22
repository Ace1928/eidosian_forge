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
def test_func_name_method_issue_479(self):
    for line, expected_name in [('o.spam(', 'spam'), ('o.spam(any([]', 'any') if pypy else ('o.spam(map([]', 'map'), ('o.spam((), ', 'spam')]:
        self.set_input_line(line)
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.current_func.__name__, expected_name)