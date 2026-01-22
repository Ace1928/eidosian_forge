import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_wildcard_with_namespace(self):
    xml = XML('<root xmlns:f="FOO"><f:foo>bar</f:foo></root>')
    self._test_eval(path='f:*', equiv='<Path "child::f:*">', input=xml, output='<foo xmlns="FOO">bar</foo>', namespaces={'f': 'FOO'})