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
def test_compare_ne(self):
    self.assertEqual(False, Expression('1 != 1').evaluate({}))
    self.assertEqual(False, Expression('x != y').evaluate({'x': 1, 'y': 1}))
    if sys.version < '3':
        self.assertEqual(False, Expression('1 <> 1').evaluate({}))
        self.assertEqual(False, Expression('x <> y').evaluate({'x': 1, 'y': 1}))