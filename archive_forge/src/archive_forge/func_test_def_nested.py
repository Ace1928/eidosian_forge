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
def test_def_nested(self):
    suite = Suite("\ndef doit():\n    values = []\n    def add(value):\n        if value not in values:\n            values.append(value)\n    add('foo')\n    add('bar')\n    return values\nx = doit()\n")
    data = {}
    suite.execute(data)
    self.assertEqual(['foo', 'bar'], data['x'])