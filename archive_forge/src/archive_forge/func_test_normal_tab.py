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
def test_normal_tab(self):
    """make sure pressing the tab key will
        still in some cases add a tab"""
    self.repl.s = ''
    self.repl.config = mock.Mock()
    self.repl.config.tab_length = 4
    self.repl.complete = mock.Mock()
    self.repl.print_line = mock.Mock()
    self.repl.tab()
    self.assertEqual(self.repl.s, '    ')