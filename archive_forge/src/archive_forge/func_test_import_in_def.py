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
def test_import_in_def(self):
    suite = Suite('def fun():\n    from itertools import repeat\n    return repeat(1, 3)\n')
    data = Context()
    suite.execute(data)
    assert 'repeat' not in data
    self.assertEqual([1, 1, 1], list(data['fun']()))