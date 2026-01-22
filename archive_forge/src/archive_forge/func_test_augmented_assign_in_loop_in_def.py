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
def test_augmented_assign_in_loop_in_def(self):
    d = {}
    Suite('def foo():\n    i = 0\n    for n in range(5):\n        i += n\n    return i\nx = foo()').execute(d)
    self.assertEqual(10, d['x'])