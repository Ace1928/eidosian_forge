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
def test_pooch_corrupted(data_dir_mirror):
    """Raise an exception if the file hash doesn't match the registry"""
    with TemporaryDirectory() as local_store:
        path = os.path.abspath(local_store)
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY_CORRUPTED)
        with capture_log() as log_file:
            with pytest.raises(ValueError) as error:
                pup.fetch('tiny-data.txt')
            assert '(tiny-data.txt)' in str(error.value)
            logs = log_file.getvalue()
            assert logs.split()[0] == 'Downloading'
            assert logs.split()[-1] == f"'{path}'."
    pup = Pooch(path=data_dir_mirror, base_url=BASEURL, registry=REGISTRY_CORRUPTED)
    with capture_log() as log_file:
        with pytest.raises(ValueError) as error:
            pup.fetch('tiny-data.txt')
        assert '(tiny-data.txt)' in str(error.value)
        logs = log_file.getvalue()
        assert logs.split()[0] == 'Updating'
        assert logs.split()[-1] == f"'{data_dir_mirror}'."