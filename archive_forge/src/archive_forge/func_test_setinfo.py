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
def test_setinfo(self):
    self.fs.create('birthday.txt')
    now = time.time()
    change_info = {'details': {'accessed': now + 60, 'modified': now + 60 * 60}}
    self.fs.setinfo('birthday.txt', change_info)
    new_info = self.fs.getinfo('birthday.txt', namespaces=['details'])
    can_write_acccess = new_info.is_writeable('details', 'accessed')
    can_write_modified = new_info.is_writeable('details', 'modified')
    if can_write_acccess:
        self.assertAlmostEqual(new_info.get('details', 'accessed'), now + 60, places=4)
    if can_write_modified:
        self.assertAlmostEqual(new_info.get('details', 'modified'), now + 60 * 60, places=4)
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.setinfo('nothing', {})