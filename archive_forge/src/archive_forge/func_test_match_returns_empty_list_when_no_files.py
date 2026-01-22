import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@mock.patch(glob_function, new=lambda text: [])
def test_match_returns_empty_list_when_no_files(self):
    self.assertEqual(self.completer.matches(2, '"a'), set())