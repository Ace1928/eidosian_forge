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
def test_def_kwargs(self):
    suite = Suite("\ndef smash(**kw):\n    return [''.join(i) for i in kw.items()]\nx = smash(foo='abc', bar='def')\n")
    data = {}
    suite.execute(data)
    self.assertEqual(['bardef', 'fooabc'], sorted(data['x']))