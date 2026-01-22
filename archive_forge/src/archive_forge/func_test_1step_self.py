import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_1step_self(self):
    xml = XML('<root><elem/></root>')
    self._test_eval(path='.', equiv='<Path "self::node()">', input=xml, output='<root><elem/></root>')
    self._test_eval(path='self::node()', equiv='<Path "self::node()">', input=xml, output='<root><elem/></root>')