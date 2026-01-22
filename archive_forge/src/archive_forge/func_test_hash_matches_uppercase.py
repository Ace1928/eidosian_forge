import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
@pytest.mark.parametrize('alg,expected_hash', list(TINY_DATA_HASHES_HASHLIB.items()), ids=list(TINY_DATA_HASHES_HASHLIB.keys()))
def test_hash_matches_uppercase(alg, expected_hash):
    """Hash matching should be independent of upper or lower case"""
    fname = os.path.join(DATA_DIR, 'tiny-data.txt')
    check_tiny_data(fname)
    known_hash = f'{alg}:{expected_hash.upper()}'
    assert hash_matches(fname, known_hash, strict=True)
    with pytest.raises(ValueError) as error:
        hash_matches(fname, known_hash[:-5], strict=True, source='Neverland')
    assert 'Neverland' in str(error.value)