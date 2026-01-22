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
@unittest.skip('disabled while non-simple completion is disabled')
def test_substring_tab_complete(self):
    self.repl.s = 'bar'
    self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.FUZZY
    self.repl.tab()
    self.assertEqual(self.repl.s, 'foobar')
    self.repl.tab()
    self.assertEqual(self.repl.s, 'foofoobar')