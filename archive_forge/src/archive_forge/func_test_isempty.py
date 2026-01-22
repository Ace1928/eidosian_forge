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
def test_isempty(self):
    self.assertTrue(self.fs.isempty('/'))
    self.fs.makedir('foo')
    self.assertFalse(self.fs.isempty('/'))
    self.assertTrue(self.fs.isempty('/foo'))
    self.fs.create('foo/bar.txt')
    self.assertFalse(self.fs.isempty('/foo'))
    self.fs.remove('foo/bar.txt')
    self.assertTrue(self.fs.isempty('/foo'))