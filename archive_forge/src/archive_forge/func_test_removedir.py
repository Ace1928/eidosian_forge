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
def test_removedir(self):
    with self.assertRaises(errors.RemoveRootError):
        self.fs.removedir('/')
    self.fs.makedirs('foo/bar/baz')
    self.assertTrue(self.fs.exists('foo/bar/baz'))
    self.fs.removedir('foo/bar/baz')
    self.assertFalse(self.fs.exists('foo/bar/baz'))
    self.assertTrue(self.fs.isdir('foo/bar'))
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.removedir('nodir')
    self.fs.makedirs('foo/bar/baz')
    self.fs.writebytes('foo/egg', b'test')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.removedir('foo/egg')
    with self.assertRaises(errors.DirectoryNotEmpty):
        self.fs.removedir('foo/bar')