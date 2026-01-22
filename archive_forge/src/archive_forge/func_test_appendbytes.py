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
def test_appendbytes(self):
    with self.assertRaises(TypeError):
        self.fs.appendbytes('foo', 'bar')
    self.fs.appendbytes('foo', b'bar')
    self.assert_bytes('foo', b'bar')
    self.fs.appendbytes('foo', b'baz')
    self.assert_bytes('foo', b'barbaz')