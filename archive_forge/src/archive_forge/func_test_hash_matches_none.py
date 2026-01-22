import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
def test_hash_matches_none():
    """The hash checking function should always returns True if known_hash=None"""
    fname = os.path.join(DATA_DIR, 'tiny-data.txt')
    assert hash_matches(fname, known_hash=None)
    assert hash_matches(fname='', known_hash=None)
    assert hash_matches(fname, known_hash=None, strict=True)