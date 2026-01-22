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
def test_class_in_def(self):
    suite = Suite("\ndef create():\n    class Foobar(object):\n        def __str__(self):\n            return 'foobar'\n    return Foobar()\nx = create()\n")
    data = {}
    suite.execute(data)
    self.assertEqual('foobar', str(data['x']))