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
def test_binop_is(self):
    self.assertEqual(True, Expression('1 is 1').evaluate({}))
    self.assertEqual(True, Expression('x is y').evaluate({'x': 1, 'y': 1}))
    self.assertEqual(False, Expression('1 is 2').evaluate({}))
    self.assertEqual(False, Expression('x is y').evaluate({'x': 1, 'y': 2}))