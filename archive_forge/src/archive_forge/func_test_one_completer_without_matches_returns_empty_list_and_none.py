import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_one_completer_without_matches_returns_empty_list_and_none(self):
    a = completer([])
    self.assertTupleEqual(autocomplete.get_completer([a], 0, ''), ([], None))