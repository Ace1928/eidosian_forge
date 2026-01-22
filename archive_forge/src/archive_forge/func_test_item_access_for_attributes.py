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
def test_item_access_for_attributes(self):

    class MyClass(object):
        myattr = 'Bar'
    data = {'mine': MyClass(), 'key': 'myattr'}
    self.assertEqual('Bar', Expression('mine.myattr').evaluate(data))
    self.assertEqual('Bar', Expression('mine["myattr"]').evaluate(data))
    self.assertEqual('Bar', Expression('mine[key]').evaluate(data))