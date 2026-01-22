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
def test_delattr(self):

    class Something(object):

        def __init__(self):
            self.attr = 'foo'
    obj = Something()
    Suite('del obj.attr').execute({'obj': obj})
    self.assertFalse(hasattr(obj, 'attr'))