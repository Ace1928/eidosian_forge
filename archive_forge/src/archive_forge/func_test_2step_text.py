import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_2step_text(self):
    xml = XML('<root><item>Foo</item></root>')
    self._test_eval(path='item/text()', equiv='<Path "child::item/child::text()">', input=xml, output='Foo')
    self._test_eval(path='*/text()', equiv='<Path "child::*/child::text()">', input=xml, output='Foo')
    self._test_eval(path='//text()', equiv='<Path "descendant-or-self::text()">', input=xml, output='Foo')
    self._test_eval(path='./text()', equiv='<Path "self::node()/child::text()">', input=xml, output='')
    xml = XML('<root><item>Foo</item><item>Bar</item></root>')
    self._test_eval(path='item/text()', equiv='<Path "child::item/child::text()">', input=xml, output='FooBar')
    xml = XML('<root><item><name>Foo</name><sub><name>Bar</name></sub></item></root>')
    self._test_eval(path='item/name/text()', equiv='<Path "child::item/child::name/child::text()">', input=xml, output='Foo')