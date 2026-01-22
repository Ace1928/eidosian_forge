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
def test_makedir(self):
    with self.assertRaises(errors.DirectoryExists):
        self.fs.makedir('/')
    slash_fs = self.fs.makedir('/', recreate=True)
    self.assertIsInstance(slash_fs, SubFS)
    self.assertEqual(self.fs.listdir('/'), [])
    self.assert_not_exists('foo')
    self.fs.makedir('foo')
    self.assert_isdir('foo')
    self.assertEqual(self.fs.gettype('foo'), ResourceType.directory)
    self.fs.writebytes('foo/bar.txt', b'egg')
    self.assert_bytes('foo/bar.txt', b'egg')
    with self.assertRaises(errors.DirectoryExists):
        self.fs.makedir('foo')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.makedir('/foo/bar/baz')
    self.fs.makedir('/foo/bar')
    self.fs.makedir('/foo/bar/baz')
    with self.assertRaises(errors.DirectoryExists):
        self.fs.makedir('foo/bar/baz')
    with self.assertRaises(errors.DirectoryExists):
        self.fs.makedir('foo/bar.txt')