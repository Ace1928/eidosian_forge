import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
@pytest.mark.skipif(paramiko is None, reason='requires paramiko to run SFTP')
def test_sftp_downloader():
    """Test sftp downloader"""
    with TemporaryDirectory() as local_store:
        downloader = SFTPDownloader(username='demo', password='password')
        url = 'sftp://test.rebex.net/pub/example/pocketftp.png'
        outfile = os.path.join(local_store, 'pocketftp.png')
        downloader(url, outfile, None)
        assert os.path.exists(outfile)