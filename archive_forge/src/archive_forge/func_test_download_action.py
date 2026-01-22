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
def test_download_action():
    """Test that the right action is performed based on file existing"""
    action, verb = download_action(Path('this_file_does_not_exist.txt'), known_hash=None)
    assert action == 'download'
    assert verb == 'Downloading'
    with temporary_file() as tmp:
        action, verb = download_action(Path(tmp), known_hash='not the correct hash')
    assert action == 'update'
    assert verb == 'Updating'
    with temporary_file() as tmp:
        with open(tmp, 'w', encoding='utf-8') as output:
            output.write('some data')
        action, verb = download_action(Path(tmp), known_hash=file_hash(tmp))
    assert action == 'fetch'
    assert verb == 'Fetching'