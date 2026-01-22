import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_only_completer_returns_None(self):
    a = completer(None)
    self.assertEqual(autocomplete.get_completer([a], 0, ''), ([], None))