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
def test_previous(self):
    self.assertEqual(self.matches_iterator.previous(), self.matches[2])
    for x in range(len(self.matches) - 1):
        self.matches_iterator.previous()
    self.assertNotEqual(self.matches_iterator.previous(), self.matches[0])
    self.assertEqual(self.matches_iterator.previous(), self.matches[1])
    self.assertEqual(self.matches_iterator.previous(), self.matches[0])