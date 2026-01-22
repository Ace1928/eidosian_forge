import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_none_returned_when_no_matches_left(self):
    com = autocomplete.DictKeyCompletion()
    local = {'d': {'ab': 1, 'cd': 2}}
    self.assertEqual(com.matches(3, 'd[r', locals_=local), None)