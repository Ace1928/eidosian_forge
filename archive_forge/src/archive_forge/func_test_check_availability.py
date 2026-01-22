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
def test_check_availability():
    """Should correctly check availability of existing and non existing files"""
    pup = Pooch(path=DATA_DIR, base_url=BASEURL, registry=REGISTRY)
    assert pup.is_available('tiny-data.txt')
    pup = Pooch(path=DATA_DIR, base_url=BASEURL + 'wrong-url/', registry=REGISTRY)
    assert not pup.is_available('tiny-data.txt')
    registry = {'not-a-real-data-file.txt': 'notarealhash'}
    registry.update(REGISTRY)
    pup = Pooch(path=DATA_DIR, base_url=BASEURL, registry=registry)
    assert not pup.is_available('not-a-real-data-file.txt')