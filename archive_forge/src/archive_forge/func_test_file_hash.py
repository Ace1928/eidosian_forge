import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
@pytest.mark.parametrize('alg,expected_hash', list(TINY_DATA_HASHES.items()), ids=list(TINY_DATA_HASHES.keys()))
def test_file_hash(alg, expected_hash):
    """Test the hash calculation using hashlib and xxhash"""
    if alg.startswith('xxh'):
        if xxhash is None:
            pytest.skip('requires xxhash')
        if alg not in ['xxh64', 'xxh32'] and XXHASH_MAJOR_VERSION < 2:
            pytest.skip('requires xxhash > 2.0')
    fname = os.path.join(DATA_DIR, 'tiny-data.txt')
    check_tiny_data(fname)
    returned_hash = file_hash(fname, alg)
    assert returned_hash == expected_hash