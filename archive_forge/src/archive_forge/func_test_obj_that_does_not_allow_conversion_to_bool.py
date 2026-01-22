import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_obj_that_does_not_allow_conversion_to_bool(self):
    com = autocomplete.DictKeyCompletion()
    local = {'mNumPy': MockNumPy()}
    self.assertEqual(com.matches(7, 'mNumPy[', locals_=local), None)