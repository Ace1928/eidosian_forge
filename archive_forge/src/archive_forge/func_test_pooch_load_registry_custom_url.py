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
def test_pooch_load_registry_custom_url():
    """Load the registry from a file with a custom URL inserted"""
    pup = Pooch(path='', base_url='')
    pup.load_registry(os.path.join(DATA_DIR, 'registry-custom-url.txt'))
    assert pup.registry == REGISTRY
    assert pup.urls == {'tiny-data.txt': 'https://some-site/tiny-data.txt'}