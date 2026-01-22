import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
def test_ftp_downloader(ftpserver):
    """Test ftp downloader"""
    with data_over_ftp(ftpserver, 'tiny-data.txt') as url:
        with TemporaryDirectory() as local_store:
            downloader = FTPDownloader(port=ftpserver.server_port)
            outfile = os.path.join(local_store, 'tiny-data.txt')
            downloader(url, outfile, None)
            check_tiny_data(outfile)