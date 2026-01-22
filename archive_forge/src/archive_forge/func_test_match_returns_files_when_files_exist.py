import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@mock.patch(glob_function, new=lambda text: ['abcde', 'aaaaa'])
@mock.patch('os.path.expanduser', new=lambda text: text)
@mock.patch('os.path.isdir', new=lambda text: False)
@mock.patch('os.path.sep', new='/')
def test_match_returns_files_when_files_exist(self):
    self.assertEqual(sorted(self.completer.matches(2, '"x')), ['aaaaa', 'abcde'])