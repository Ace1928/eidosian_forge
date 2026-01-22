import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
def test_copy_attributes(self):
    self.assertEqual(self._apply('foo/@*', with_attrs=True)[1], [[(None, ATTR, {u'name': u'foo', u'size': u'100'})]])