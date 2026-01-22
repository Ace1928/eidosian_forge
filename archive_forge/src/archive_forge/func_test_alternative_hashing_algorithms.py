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
def test_alternative_hashing_algorithms(data_dir_mirror):
    """Test different hashing algorithms using local data"""
    fname = str(data_dir_mirror / 'tiny-data.txt')
    check_tiny_data(fname)
    with open(fname, 'rb') as fin:
        data = fin.read()
    for alg in ('sha512', 'md5'):
        hasher = hashlib.new(alg)
        hasher.update(data)
        registry = {'tiny-data.txt': f'{alg}:{hasher.hexdigest()}'}
        pup = Pooch(path=data_dir_mirror, base_url='some bogus URL', registry=registry)
        assert fname == pup.fetch('tiny-data.txt')
        check_tiny_data(fname)