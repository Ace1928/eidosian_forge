from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
def test_movedir(self):
    self.fs.makedirs('foo/bar/baz/egg')
    self.fs.writetext('foo/bar/foofoo.txt', 'Hello')
    self.fs.makedir('foo2')
    self.fs.movedir('foo/bar', 'foo2')
    self.assert_text('foo2/foofoo.txt', 'Hello')
    self.assert_isdir('foo2/baz/egg')
    self.assert_not_exists('foo/bar')
    self.assert_not_exists('foo/bar/foofoo.txt')
    self.assert_not_exists('foo/bar/baz/egg')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.movedir('foo', 'foofoo')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.movedir('spam', 'egg', create=True)
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.movedir('foo2/foofoo.txt', 'foo2/baz/egg')