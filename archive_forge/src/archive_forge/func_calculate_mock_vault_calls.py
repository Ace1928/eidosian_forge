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
def calculate_mock_vault_calls(data, part_size, chunk_size):
    upload_part_calls = []
    data_tree_hashes = []
    for i, data_part in enumerate(partify(data, part_size)):
        start = i * part_size
        end = start + len(data_part)
        data_part_tree_hash_blob = tree_hash(chunk_hashes(data_part, chunk_size))
        data_part_tree_hash = bytes_to_hex(data_part_tree_hash_blob)
        data_part_linear_hash = sha256(data_part).hexdigest()
        upload_part_calls.append(call(sentinel.vault_name, sentinel.upload_id, data_part_linear_hash, data_part_tree_hash, (start, end - 1), data_part))
        data_tree_hashes.append(data_part_tree_hash_blob)
    return (upload_part_calls, data_tree_hashes)