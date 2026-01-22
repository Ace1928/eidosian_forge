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
def test_call_keywords(self):
    self.assertEqual(42, Expression('foo(x=bar)').evaluate({'foo': lambda x: x, 'bar': 42}))