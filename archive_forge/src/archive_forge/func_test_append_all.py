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
def test_append_all(self):
    self.assertEqual(self._apply('*|text()'), [(None, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (None, TEXT, u'CONTENT 1'), (EXIT, END, u'foo'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (None, TEXT, u'CONTENT 2'), (EXIT, END, u'bar'), (None, END, u'root')])