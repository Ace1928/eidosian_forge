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
def test_writetext(self):
    self.fs.writetext('foo', 'bar')
    with self.fs.open('foo', 'rt') as f:
        foo = f.read()
    self.assertEqual(foo, 'bar')
    self.assertIsInstance(foo, text_type)
    with self.assertRaises(TypeError):
        self.fs.writetext('nottext', b'bytes')