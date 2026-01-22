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
def test_current_function(self):
    self.set_input_line('INPUTLINE')
    self.repl.current_func = inspect.getsource
    self.assertIn('text of the source code', self.repl.get_source_of_current_name())
    self.assert_get_source_error_for_current_function([], 'No source code found for INPUTLINE')
    self.assert_get_source_error_for_current_function(list.pop, 'No source code found for INPUTLINE')