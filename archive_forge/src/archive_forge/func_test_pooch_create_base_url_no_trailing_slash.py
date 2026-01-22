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
def test_pooch_create_base_url_no_trailing_slash():
    """
    Test if pooch.create appends a trailing slash to the base url if missing
    """
    base_url = 'https://mybase.url'
    pup = create(base_url=base_url, registry=None, path=DATA_DIR)
    assert pup.base_url == base_url + '/'