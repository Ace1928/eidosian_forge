import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_locate_fails_when_not_in_string(self):
    self.assertEqual(self.completer.locate(4, 'abcd'), None)