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
def test_for_in_def(self):
    suite = Suite('def loop():\n    for i in range(10):\n        if i == 5:\n            break\n    return i\n')
    data = {}
    suite.execute(data)
    assert 'loop' in data
    self.assertEqual(5, data['loop']())