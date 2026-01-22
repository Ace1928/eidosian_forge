import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_child_position(self):
    xml = XML('<root><a><b>1</b><b>2</b><b>3</b></a><a><b>4</b><b>5</b></a></root>')
    self._test_eval('//a/b[2]', input=xml, output='<b>2</b><b>5</b>')
    self._test_eval('//a/b[3]', input=xml, output='<b>3</b>')