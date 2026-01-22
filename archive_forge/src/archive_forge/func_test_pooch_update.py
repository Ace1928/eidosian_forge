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
@pytest.mark.network
def test_pooch_update():
    """Setup a pooch that already has the local data but the file is outdated"""
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        true_path = str(path / 'tiny-data.txt')
        with open(true_path, 'w', encoding='utf-8') as fin:
            fin.write('different data')
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY)
        with capture_log() as log_file:
            fname = pup.fetch('tiny-data.txt')
            logs = log_file.getvalue()
            assert logs.split()[0] == 'Updating'
            assert logs.split()[-1] == f"'{path}'."
        assert true_path == fname
        check_tiny_data(fname)
        assert file_hash(fname) == REGISTRY['tiny-data.txt']
        with capture_log() as log_file:
            fname = pup.fetch('tiny-data.txt')
            assert log_file.getvalue() == ''