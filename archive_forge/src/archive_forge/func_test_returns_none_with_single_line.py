import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_returns_none_with_single_line(self):
    com = autocomplete.MultilineJediCompletion()
    self.assertEqual(com.matches(2, 'Va', current_block='Va', history=[]), None)