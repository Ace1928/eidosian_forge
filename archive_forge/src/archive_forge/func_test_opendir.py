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
def test_opendir(self):
    self.fs.makedir('foo')
    self.fs.writebytes('foo/bar', b'barbar')
    self.fs.writebytes('foo/egg', b'eggegg')
    with self.fs.opendir('foo') as foo_fs:
        repr(foo_fs)
        text_type(foo_fs)
        six.assertCountEqual(self, foo_fs.listdir('/'), ['bar', 'egg'])
        self.assertTrue(foo_fs.isfile('bar'))
        self.assertTrue(foo_fs.isfile('egg'))
        self.assertEqual(foo_fs.readbytes('bar'), b'barbar')
        self.assertEqual(foo_fs.readbytes('egg'), b'eggegg')
    self.assertFalse(self.fs.isclosed())
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.opendir('egg')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.opendir('foo/egg')
    self.fs.opendir('')
    self.fs.opendir('/')
    with self.fs.opendir('foo', factory=ClosingSubFS) as foo_fs:
        six.assertCountEqual(self, foo_fs.listdir('/'), ['bar', 'egg'])
        self.assertTrue(foo_fs.isfile('bar'))
        self.assertTrue(foo_fs.isfile('egg'))
        self.assertEqual(foo_fs.readbytes('bar'), b'barbar')
        self.assertEqual(foo_fs.readbytes('egg'), b'eggegg')
    self.assertTrue(self.fs.isclosed())