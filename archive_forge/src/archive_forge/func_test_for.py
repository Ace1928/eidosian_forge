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
def test_for(self):
    suite = Suite('x = []\nfor i in range(3):\n    x.append(i**2)\n')
    data = {}
    suite.execute(data)
    self.assertEqual([0, 1, 4], data['x'])