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
@pytest.fixture
def data_dir_mirror(tmp_path):
    """
    Mirror the test data folder on a temporary directory. Needed to avoid
    permission errors when pooch is installed on a non-writable path.
    """
    return mirror_directory(DATA_DIR, tmp_path)