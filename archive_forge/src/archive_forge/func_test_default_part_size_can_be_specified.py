import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_default_part_size_can_be_specified(self):
    default_part_size = 2 * 1024 * 1024
    self.assertEqual(minimum_part_size(8 * 1024 * 1024, default_part_size), default_part_size)