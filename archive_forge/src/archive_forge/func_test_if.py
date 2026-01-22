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
def test_if(self):
    suite = Suite('if foo == 42:\n    x = True\n')
    data = {'foo': 42}
    suite.execute(data)
    self.assertEqual(True, data['x'])