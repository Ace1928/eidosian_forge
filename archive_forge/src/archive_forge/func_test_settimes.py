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
def test_settimes(self):
    self.fs.create('birthday.txt')
    self.fs.settimes('birthday.txt', accessed=datetime(2016, 7, 5))
    info = self.fs.getinfo('birthday.txt', namespaces=['details'])
    can_write_acccess = info.is_writeable('details', 'accessed')
    can_write_modified = info.is_writeable('details', 'modified')
    if can_write_acccess:
        self.assertEqual(info.accessed, datetime(2016, 7, 5, tzinfo=timezone.utc))
    if can_write_modified:
        self.assertEqual(info.modified, datetime(2016, 7, 5, tzinfo=timezone.utc))