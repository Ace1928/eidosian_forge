import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_3step_complex(self):
    self._test_eval(path='*/bar', equiv='<Path "child::*/child::bar">', input=XML('<root><foo><bar/></foo></root>'), output='<bar/>')
    self._test_eval(path='//bar', equiv='<Path "descendant-or-self::bar">', input=XML('<root><foo><bar id="1"/></foo><bar id="2"/></root>'), output='<bar id="1"/><bar id="2"/>')