import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_no_completers_fails(self):
    with self.assertRaises(ValueError):
        autocomplete.CumulativeCompleter([])