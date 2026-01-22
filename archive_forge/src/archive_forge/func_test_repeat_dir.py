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
def test_repeat_dir(self):
    self.fs.makedirs('foo/foo/foo')
    self.assertEqual(self.fs.listdir(''), ['foo'])
    self.assertEqual(self.fs.listdir('foo'), ['foo'])
    self.assertEqual(self.fs.listdir('foo/foo'), ['foo'])
    self.assertEqual(self.fs.listdir('foo/foo/foo'), [])
    scan = list(self.fs.scandir('foo'))
    self.assertEqual(len(scan), 1)
    self.assertEqual(scan[0].name, 'foo')