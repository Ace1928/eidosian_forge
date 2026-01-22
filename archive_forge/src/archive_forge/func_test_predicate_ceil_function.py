import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_ceil_function(self):
    xml = XML('<root><foo>bar</foo></root>')
    self._test_eval('*[ceiling("4.5")=5]', input=xml, output='<foo>bar</foo>')