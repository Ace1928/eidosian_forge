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
def test_unaryop_pos(self):
    self.assertEqual(1, Expression('+1').evaluate({}))
    self.assertEqual(1, Expression('+x').evaluate({'x': 1}))