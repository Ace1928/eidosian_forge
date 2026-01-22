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
def test_retrieve_default_path():
    """Try downloading some data with retrieve to the default cache location"""
    data_file = 'tiny-data.txt'
    url = BASEURL + data_file
    expected_location = os_cache('pooch') / data_file
    try:
        with capture_log() as log_file:
            fname = retrieve(url, known_hash=None, fname=data_file)
            logs = log_file.getvalue()
            assert logs.split()[0] == 'Downloading'
            assert str(os_cache('pooch').resolve()) in logs
            assert 'SHA256 hash of downloaded file' in logs
            assert REGISTRY[data_file] in logs
        assert fname == str(expected_location.resolve())
        check_tiny_data(fname)
        assert file_hash(fname) == REGISTRY[data_file]
    finally:
        if os.path.exists(str(expected_location)):
            os.remove(str(expected_location))