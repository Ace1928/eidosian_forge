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
def test_finally(self):
    suite = Suite('try:\n    x = 2\nfinally:\n    x = None\n')
    data = {}
    suite.execute(data)
    self.assertEqual(None, data['x'])