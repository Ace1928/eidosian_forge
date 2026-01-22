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
def test_autospec_side_effect_exception(self):

    def f():
        pass
    mock = create_autospec(f)
    mock.side_effect = ValueError('Bazinga!')
    self.assertRaisesRegex(ValueError, 'Bazinga!', mock)