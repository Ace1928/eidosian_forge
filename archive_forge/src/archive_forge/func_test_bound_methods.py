from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
@unittest.skipIf('PyPy' in sys.version, 'This fails differently on pypy')
def test_bound_methods(self):
    m = Mock()
    m.__iter__ = [3].__iter__
    self.assertRaises(TypeError, iter, m)