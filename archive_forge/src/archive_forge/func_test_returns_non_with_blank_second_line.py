import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_returns_non_with_blank_second_line(self):
    com = autocomplete.MultilineJediCompletion()
    self.assertEqual(com.matches(0, '', current_block='class Foo():\n', history=['class Foo():']), None)