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
def test_binop_contains(self):
    self.assertEqual(True, Expression('1 in (1, 2, 3)').evaluate({}))
    self.assertEqual(True, Expression('x in y').evaluate({'x': 1, 'y': (1, 2, 3)}))