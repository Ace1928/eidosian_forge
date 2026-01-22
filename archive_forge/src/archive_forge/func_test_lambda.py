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
def test_lambda(self):
    data = {'items': range(5)}
    expr = Expression('filter(lambda x: x > 2, items)')
    self.assertEqual([3, 4], list(expr.evaluate(data)))