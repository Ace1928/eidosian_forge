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
def test_simple_attribute_complete(self):
    self.repl = FakeRepl({'autocomplete_mode': autocomplete.AutocompleteModes.SIMPLE})
    self.set_input_line('Foo.b')
    code = 'class Foo():\n\tdef bar(self):\n\t\tpass\n'
    for line in code.split('\n'):
        self.repl.push(line)
    self.assertTrue(self.repl.complete())
    self.assertTrue(hasattr(self.repl.matches_iter, 'matches'))
    self.assertEqual(self.repl.matches_iter.matches, ['Foo.bar'])