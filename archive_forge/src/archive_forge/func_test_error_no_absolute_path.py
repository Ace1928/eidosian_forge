import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_error_no_absolute_path(self):
    self.assertRaises(PathSyntaxError, Path, '/root')