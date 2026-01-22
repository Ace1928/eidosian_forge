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
def test_pooch_update_disallowed_environment():
    """Test that disallowing updates works through an environment variable."""
    variable_name = 'MYPROJECT_DISALLOW_UPDATES'
    try:
        os.environ[variable_name] = 'False'
        with TemporaryDirectory() as local_store:
            path = Path(local_store)
            true_path = str(path / 'tiny-data.txt')
            with open(true_path, 'w', encoding='utf-8') as fin:
                fin.write('different data')
            pup = create(path=path, base_url=BASEURL, registry=REGISTRY, allow_updates=variable_name)
            with pytest.raises(ValueError):
                pup.fetch('tiny-data.txt')
    finally:
        os.environ.pop(variable_name)