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
def test_download_chunk_size(self):
    test_bytes = b'Hello, World' * 100
    self.fs.writebytes('hello.bin', test_bytes)
    write_file = io.BytesIO()
    self.fs.download('hello.bin', write_file, chunk_size=8)
    self.assertEqual(write_file.getvalue(), test_bytes)