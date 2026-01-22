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
def test_slice_attribute(self):

    class ValueHolder:

        def __init__(self):
            self.value = 3
    suite = Suite('x = numbers[obj.value]')
    data = {'numbers': [0, 1, 2, 3, 4], 'obj': ValueHolder()}
    suite.execute(data)
    self.assertEqual(3, data['x'])