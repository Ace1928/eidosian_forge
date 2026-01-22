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
def test_yield_expression(self):
    d = {}
    suite = Suite('\nresults = []\ndef counter(maximum):\n    i = 0\n    while i < maximum:\n        val = (yield i)\n        if val is not None:\n            i = val\n        else:\n            i += 1\nit = counter(5)\nresults.append(next(it))\nresults.append(it.send(3))\nresults.append(next(it))\n')
    suite.execute(d)
    self.assertEqual([0, 3, 4], d['results'])