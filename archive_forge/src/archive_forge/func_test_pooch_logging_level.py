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
def test_pooch_logging_level():
    """Setup a pooch and check that no logging happens when the level is raised"""
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        urls = {'tiny-data.txt': BASEURL + 'tiny-data.txt'}
        pup = Pooch(path=path, base_url='', registry=REGISTRY, urls=urls)
        with capture_log('CRITICAL') as log_file:
            fname = pup.fetch('tiny-data.txt')
            assert log_file.getvalue() == ''
        check_tiny_data(fname)