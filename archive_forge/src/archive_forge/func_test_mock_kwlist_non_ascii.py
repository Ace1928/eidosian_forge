import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_mock_kwlist_non_ascii(self):
    with mock.patch.object(keyword, 'kwlist', new=['abcß']):
        self.assertEqual(self.com.matches(3, 'abc', locals_={}), None)