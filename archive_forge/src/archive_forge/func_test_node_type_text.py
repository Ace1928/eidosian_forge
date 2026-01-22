import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_node_type_text(self):
    xml = XML('<root>Some text <br/>in here.</root>')
    self._test_eval(path='text()', equiv='<Path "child::text()">', input=xml, output='Some text in here.')