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
def test_openbin(self):
    with self.fs.openbin('file.bin', 'wb') as write_file:
        repr(write_file)
        text_type(write_file)
        self.assertIn('b', write_file.mode)
        self.assertIsInstance(write_file, io.IOBase)
        self.assertTrue(write_file.writable())
        self.assertFalse(write_file.readable())
        self.assertFalse(write_file.closed)
        self.assertEqual(3, write_file.write(b'\x00\x01\x02'))
    self.assertTrue(write_file.closed)
    with self.fs.openbin('file.bin', 'rb') as read_file:
        repr(write_file)
        text_type(write_file)
        self.assertIn('b', read_file.mode)
        self.assertIsInstance(read_file, io.IOBase)
        self.assertTrue(read_file.readable())
        self.assertFalse(read_file.writable())
        self.assertFalse(read_file.closed)
        data = read_file.read()
    self.assertEqual(data, b'\x00\x01\x02')
    self.assertTrue(read_file.closed)
    with self.assertRaises(ValueError):
        with self.fs.openbin('file.bin', 'rt') as read_file:
            pass
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('foo.bin')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('/foo/bar/test.txt')
    self.fs.makedir('foo')
    with self.assertRaises(errors.FileExpected):
        self.fs.openbin('/foo')
    with self.assertRaises(errors.FileExpected):
        self.fs.openbin('/foo', 'w')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('/egg/bar')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('/egg/bar', 'w')
    with self.assertRaises(ValueError):
        self.fs.openbin('foo.bin', 'h')