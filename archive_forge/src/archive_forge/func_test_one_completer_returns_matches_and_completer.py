import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_one_completer_returns_matches_and_completer(self):
    a = completer(['a'])
    self.assertTupleEqual(autocomplete.get_completer([a], 0, ''), (['a'], a))