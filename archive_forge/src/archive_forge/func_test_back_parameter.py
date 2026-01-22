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
def test_back_parameter(self):
    self.repl.matches_iter = mock.Mock()
    self.repl.matches_iter.matches = True
    self.repl.matches_iter.previous.return_value = 'previtem'
    self.repl.matches_iter.is_cseq.return_value = False
    self.repl.show_list = mock.Mock()
    self.repl.funcprops = mock.Mock()
    self.repl.arg_pos = mock.Mock()
    self.repl.matches_iter.cur_line.return_value = (None, 'previtem')
    self.repl.print_line = mock.Mock()
    self.repl.s = 'foo'
    self.repl.cpos = 0
    self.repl.tab(back=True)
    self.assertTrue(self.repl.matches_iter.previous.called)
    self.assertTrue(self.repl.s, 'previtem')