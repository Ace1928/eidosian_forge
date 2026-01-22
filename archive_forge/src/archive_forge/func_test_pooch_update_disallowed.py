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
def test_pooch_update_disallowed():
    """Test that disallowing updates works."""
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        true_path = str(path / 'tiny-data.txt')
        with open(true_path, 'w', encoding='utf-8') as fin:
            fin.write('different data')
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY, allow_updates=False)
        with pytest.raises(ValueError):
            pup.fetch('tiny-data.txt')