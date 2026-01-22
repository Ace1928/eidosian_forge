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
def test_assign_to_attribute(self):

    class Something(object):
        pass
    something = Something()
    suite = Suite("obj.attr = 'foo'")
    data = {'obj': something}
    suite.execute(data)
    self.assertEqual('foo', something.attr)