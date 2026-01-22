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
def test_wrap_adjacent_elements(self):
    self.assertEqual(_transform(FOOBAR, Transformer('foo|bar').wrap('wrap')), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'wrap'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, END, u'wrap'), (None, START, u'wrap'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, END, u'wrap'), (None, END, u'root')])