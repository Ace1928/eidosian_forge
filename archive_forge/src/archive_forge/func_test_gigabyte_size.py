import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_gigabyte_size(self):
    self.assertEqual(minimum_part_size(8 * 1024 * 1024 * 10000), 8 * 1024 * 1024)