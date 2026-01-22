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
def test_filter_after_outside(self):
    stream = _transform('<root>x</root>', Transformer('//root/text()').filter(lambda x: x))
    self.assertEqual(list(stream), [(None, START, u'root'), (OUTSIDE, TEXT, u'x'), (None, END, u'root')])