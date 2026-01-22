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
def test_islink(self):
    self.fs.touch('foo')
    self.assertFalse(self.fs.islink('foo'))
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.islink('bar')