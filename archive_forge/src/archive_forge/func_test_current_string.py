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
def test_current_string(self):
    self.set_input_line('a = "2"')
    self.repl.cpos = 0
    self.assertEqual(self.repl.current_string(), '"2"')
    self.set_input_line('a = "2" + 2')
    self.assertEqual(self.repl.current_string(), '')