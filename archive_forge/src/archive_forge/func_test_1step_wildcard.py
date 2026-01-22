import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_1step_wildcard(self):
    xml = XML('<root><elem/></root>')
    self._test_eval(path='*', equiv='<Path "child::*">', input=xml, output='<elem/>')
    self._test_eval(path='child::*', equiv='<Path "child::*">', input=xml, output='<elem/>')
    self._test_eval(path='child::node()', equiv='<Path "child::node()">', input=xml, output='<elem/>')
    self._test_eval(path='//*', equiv='<Path "descendant-or-self::*">', input=xml, output='<root><elem/></root>')