from hashlib import sha256
import itertools
from boto.compat import StringIO
from tests.unit import unittest
from mock import (
from nose.tools import assert_equal
from boto.glacier.layer1 import Layer1
from boto.glacier.vault import Vault
from boto.glacier.writer import Writer, resume_file_upload
from boto.glacier.utils import bytes_to_hex, chunk_hashes, tree_hash
def test_current_tree_hash(self):
    self.writer.write(b'1234')
    self.writer.write(b'567')
    hash_1 = self.writer.current_tree_hash
    self.assertEqual(hash_1, b'\x0e\xb0\x11Z\x1d\x1f\n\x10|\xf76\xa6\xf5' + b'\x83\xd1\xd5"bU\x0c\x95\xa8<\xf5\x81\xef\x0e\x0f\x95\n\xb7k')
    self.writer.write(b'22i3uy')
    hash_2 = self.writer.current_tree_hash
    self.assertEqual(hash_2, b'\x7f\xf4\x97\x82U]\x81R\x05#^\xe8\x1c\xd19' + b'\xe8\x1f\x9e\xe0\x1aO\xaad\xe5\x06"\xa5\xc0\xa8AdL')
    self.writer.close()
    final_hash = self.writer.current_tree_hash
    self.assertEqual(final_hash, b';\x1a\xb8!=\xf0\x14#\x83\x11\xd5\x0b\x0f' + b'\xc7D\xe4\x8e\xd1W\x99z\x14\x06\xb9D\xd0\xf0*\x93\xa2\x8e\xf9')
    self.assertEqual(final_hash, self.writer.current_tree_hash)