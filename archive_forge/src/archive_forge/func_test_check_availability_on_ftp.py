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
def test_check_availability_on_ftp(ftpserver):
    """Should correctly check availability of existing and non existing files"""
    with data_over_ftp(ftpserver, 'tiny-data.txt') as url:
        pup = Pooch(path=DATA_DIR, base_url=url.replace('tiny-data.txt', ''), registry={'tiny-data.txt': 'baee0894dba14b12085eacb204284b97e362f4f3e5a5807693cc90ef415c1b2d', 'doesnot_exist.zip': 'jdjdjdjdflld'})
        downloader = FTPDownloader(port=ftpserver.server_port)
        assert pup.is_available('tiny-data.txt', downloader=downloader)
        assert not pup.is_available('doesnot_exist.zip', downloader=downloader)