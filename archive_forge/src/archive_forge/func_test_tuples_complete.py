import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_tuples_complete(self):
    self.assertSetEqual(self.com.matches(5, 'a[0].', locals_={'a': (Foo(),)}), {'method', 'a', 'b'})