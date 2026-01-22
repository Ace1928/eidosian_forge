import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_none_returned_when_eval_error(self):
    com = autocomplete.DictKeyCompletion()
    local = {'e': {'ab': 1, 'cd': 2}}
    self.assertEqual(com.matches(2, 'd[', locals_=local), None)