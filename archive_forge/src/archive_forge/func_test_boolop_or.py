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
def test_boolop_or(self):
    self.assertEqual(True, Expression('True or False').evaluate({}))
    self.assertEqual(True, Expression('x or y').evaluate({'x': True, 'y': False}))