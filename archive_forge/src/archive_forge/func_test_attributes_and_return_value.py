from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_attributes_and_return_value(self):
    mock = MagicMock()
    attr = mock.foo

    def _get_type(obj):
        return type(obj).__mro__[1]
    self.assertEqual(_get_type(attr), MagicMock)
    returned = mock()
    self.assertEqual(_get_type(returned), MagicMock)