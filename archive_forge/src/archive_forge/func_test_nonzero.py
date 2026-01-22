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
def test_nonzero(self):
    """self.matches_iterator should be False at start,
        then True once we active a match.
        """
    self.assertFalse(self.matches_iterator)
    next(self.matches_iterator)
    self.assertTrue(self.matches_iterator)