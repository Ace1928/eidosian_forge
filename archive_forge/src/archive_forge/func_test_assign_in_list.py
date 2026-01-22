import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_assign_in_list(self):
    suite = Suite("[d['k']] = 'foo',; assert d['k'] == 'foo'")
    d = {'k': 'bar'}
    suite.execute({'d': d})
    self.assertEqual('foo', d['k'])