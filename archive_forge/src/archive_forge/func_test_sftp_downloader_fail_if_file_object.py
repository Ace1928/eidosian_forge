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
def test_sftp_downloader_fail_if_file_object():
    """Downloader should fail when a file object rather than string is passed"""
    with TemporaryDirectory() as local_store:
        downloader = SFTPDownloader(username='demo', password='password')
        url = 'sftp://test.rebex.net/pub/example/pocketftp.png'
        outfile = os.path.join(local_store, 'pocketftp.png')
        with open(outfile, 'wb') as outfile_obj:
            with pytest.raises(TypeError):
                downloader(url, outfile_obj, None)