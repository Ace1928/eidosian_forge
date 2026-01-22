import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_autospec_side_effect(self):
    results = [1, 2, 3]

    def effect():
        return results.pop()

    def f():
        pass
    mock = create_autospec(f)
    mock.side_effect = [1, 2, 3]
    self.assertEqual([mock(), mock(), mock()], [1, 2, 3], 'side effect not used correctly in create_autospec')
    results = [1, 2, 3]
    mock = create_autospec(f)
    mock.side_effect = effect
    self.assertEqual([mock(), mock(), mock()], [3, 2, 1], 'callable side effect not used correctly')