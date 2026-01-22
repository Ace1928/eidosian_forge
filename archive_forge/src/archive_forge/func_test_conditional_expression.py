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
def test_conditional_expression(self):
    expr = Expression("'T' if foo else 'F'")
    self.assertEqual('T', expr.evaluate({'foo': True}))
    self.assertEqual('F', expr.evaluate({'foo': False}))