import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_attr_selection(self):
    xml = XML('<root><foo bar="abc"></foo></root>')
    path = Path('foo/@bar')
    result = path.select(xml)
    self.assertEqual(list(result), [Attrs([(QName('bar'), u'abc')])])