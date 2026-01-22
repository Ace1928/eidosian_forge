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
def test_binop_or(self):
    self.assertEqual(1, Expression('1 | 0').evaluate({}))
    self.assertEqual(1, Expression('x | y').evaluate({'x': 1, 'y': 0}))