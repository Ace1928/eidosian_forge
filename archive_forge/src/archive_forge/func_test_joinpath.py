import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_joinpath(self):

    def test(expected, *args):
        joined = urlutils.joinpath(*args)
        self.assertEqual(expected, joined)
    test('foo', 'foo')
    test('foo/bar', 'foo', 'bar')
    test('foo/bar', 'foo', '.', 'bar')
    test('foo/baz', 'foo', 'bar', '../baz')
    test('foo/bar/baz', 'foo', 'bar/baz')
    test('foo/baz', 'foo', 'bar/../baz')
    test('/foo', '/foo')
    test('/foo', '/foo', '.')
    test('/foo/bar', '/foo', 'bar')
    test('/', '/foo', '..')
    test('/bar', 'foo', '/bar')
    test('foo/bar', 'foo/', 'bar')
    self.assertRaises(urlutils.InvalidURLJoin, urlutils.joinpath, '/', '../baz')
    self.assertRaises(urlutils.InvalidURLJoin, urlutils.joinpath, '/', '..')
    self.assertRaises(urlutils.InvalidURLJoin, urlutils.joinpath, '/', '/..')