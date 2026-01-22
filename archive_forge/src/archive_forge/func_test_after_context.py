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
def test_after_context(self):
    self.assertEqual(self._apply('.'), [(ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (INSIDE, START, u'bar'), (INSIDE, TEXT, u'BAR'), (INSIDE, END, u'bar'), (EXIT, END, u'root'), (None, TEXT, u'CONTENT 1')])