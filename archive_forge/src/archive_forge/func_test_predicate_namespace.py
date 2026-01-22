import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_namespace(self):
    xml = XML('<root><foo xmlns="NS"/><bar/></root>')
    self._test_eval('*[namespace-uri()="NS"]', input=xml, output='<foo xmlns="NS"/>')