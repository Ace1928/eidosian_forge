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
def test_getitem_with_constant_string(self):
    data = dict(dict={'some': 'thing'})
    self.assertEqual('thing', Expression("dict['some']").evaluate(data))