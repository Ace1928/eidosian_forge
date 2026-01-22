import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
@unittest.skipUnless(six.PY3, 'Python 3 requires reading binary!')
def test_compute_hash_tempfile_py3(self):
    with tempfile.TemporaryFile(mode='w+') as f:
        with self.assertRaises(ValueError):
            compute_hashes_from_fileobj(f, chunk_size=512)
    f = StringIO('test data' * 500)
    compute_hashes_from_fileobj(f, chunk_size=512)