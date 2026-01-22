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
@pytest.mark.parametrize('fname', ['tiny-data.txt', 'subdir/tiny-data.txt'])
def test_stream_download(fname):
    """Check that downloading a file over HTTP works as expected"""
    url = BASEURL + 'store/' + fname
    known_hash = REGISTRY[fname]
    downloader = HTTPDownloader()
    with TemporaryDirectory() as local_store:
        destination = Path(local_store) / fname
        assert not destination.exists()
        stream_download(url, destination, known_hash, downloader, pooch=None)
        assert destination.exists()
        check_tiny_data(str(destination))