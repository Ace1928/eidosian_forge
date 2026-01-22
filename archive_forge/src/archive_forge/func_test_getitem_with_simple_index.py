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
def test_getitem_with_simple_index(self):
    data = dict(values={True: 'bar', 2.5: 'baz', None: 'quox', 42: 'quooox', b'foo': 'foobar'})
    self.assertEqual('bar', Expression('values[True]').evaluate(data))
    self.assertEqual('baz', Expression('values[2.5]').evaluate(data))
    self.assertEqual('quooox', Expression('values[42]').evaluate(data))
    self.assertEqual('foobar', Expression('values[b"foo"]').evaluate(data))
    self.assertEqual('quox', Expression('values[None]').evaluate(data))