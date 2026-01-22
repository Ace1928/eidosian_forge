import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_self_and_descendant(self):
    xml = XML('<root><foo/></root>')
    self._test_eval('self::root', input=xml, output='<root><foo/></root>')
    self._test_eval('self::foo', input=xml, output='')
    self._test_eval('descendant::root', input=xml, output='')
    self._test_eval('descendant::foo', input=xml, output='<foo/>')
    self._test_eval('descendant-or-self::root', input=xml, output='<root><foo/></root>')
    self._test_eval('descendant-or-self::foo', input=xml, output='<foo/>')