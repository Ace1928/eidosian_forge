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
def test_tuple_literal(self):
    self.assertEqual((), Expression('()').evaluate({}))
    self.assertEqual((1, 2, 3), Expression('(1, 2, 3)').evaluate({}))
    self.assertEqual((True,), Expression('(value,)').evaluate({'value': True}))