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
def test_move_file_same_fs(self):
    text = 'Hello, World'
    self.fs.makedir('foo').writetext('test.txt', text)
    self.assert_text('foo/test.txt', text)
    fs.move.move_file(self.fs, 'foo/test.txt', self.fs, 'foo/test2.txt')
    self.assert_not_exists('foo/test.txt')
    self.assert_text('foo/test2.txt', text)
    self.assertEqual(self.fs.listdir('foo'), ['test2.txt'])
    self.assertEqual(next(self.fs.scandir('foo')).name, 'test2.txt')