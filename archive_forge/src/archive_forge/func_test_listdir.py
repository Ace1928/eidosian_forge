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
def test_listdir(self):
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.listdir('foobar')
    self.assertEqual(self.fs.listdir('/'), [])
    self.assertEqual(self.fs.listdir('.'), [])
    self.assertEqual(self.fs.listdir('./'), [])
    self.fs.writebytes('foo', b'egg')
    self.fs.writebytes('bar', b'egg')
    self.fs.makedir('baz')
    self.fs.writebytes('baz/egg', b'egg')
    six.assertCountEqual(self, self.fs.listdir('/'), ['foo', 'bar', 'baz'])
    six.assertCountEqual(self, self.fs.listdir('.'), ['foo', 'bar', 'baz'])
    six.assertCountEqual(self, self.fs.listdir('./'), ['foo', 'bar', 'baz'])
    for name in self.fs.listdir('/'):
        self.assertIsInstance(name, text_type)
    self.fs.makedir('dir')
    self.assertEqual(self.fs.listdir('/dir'), [])
    self.fs.writebytes('dir/foofoo', b'egg')
    self.fs.writebytes('dir/barbar', b'egg')
    six.assertCountEqual(self, self.fs.listdir('dir'), ['foofoo', 'barbar'])
    for name in self.fs.listdir('dir'):
        self.assertIsInstance(name, text_type)
    self.fs.create('notadir')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.listdir('notadir')