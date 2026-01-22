import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_1step_attribute(self):
    self._test_eval(path='@foo', equiv='<Path "attribute::foo">', input=XML('<root/>'), output='')
    xml = XML('<root foo="bar"/>')
    self._test_eval(path='@foo', equiv='<Path "attribute::foo">', input=xml, output='bar')
    self._test_eval(path='./@foo', equiv='<Path "self::node()/attribute::foo">', input=xml, output='bar')