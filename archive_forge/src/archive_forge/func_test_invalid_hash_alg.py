import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from ..core import create, Pooch, retrieve, download_action, stream_download
from ..utils import get_logger, temporary_file, os_cache
from ..hashes import file_hash, hash_matches
from .. import core
from ..downloaders import HTTPDownloader, FTPDownloader
from .utils import (
def test_invalid_hash_alg(data_dir_mirror):
    """Test an invalid hashing algorithm"""
    pup = Pooch(path=data_dir_mirror, base_url=BASEURL, registry={'tiny-data.txt': 'blah:1234'})
    with pytest.raises(ValueError) as exc:
        pup.fetch('tiny-data.txt')
    assert "'blah'" in str(exc.value)