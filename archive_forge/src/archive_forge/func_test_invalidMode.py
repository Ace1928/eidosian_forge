import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_invalidMode(self):
    """
        A ChunkingZipFile opened in write-mode should not allow .readfile(),
        and raise a RuntimeError instead.
        """
    with zipstream.ChunkingZipFile(self.mktemp(), 'w') as czf:
        self.assertRaises(RuntimeError, czf.readfile, 'something')