import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_attribute(self):
    self.assertEqual(autocomplete._after_last_dot('abc.edf'), 'edf')