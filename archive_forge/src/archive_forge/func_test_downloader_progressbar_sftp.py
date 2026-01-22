import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
@pytest.mark.skipif(tqdm is None, reason='requires tqdm')
@pytest.mark.skipif(paramiko is None, reason='requires paramiko')
def test_downloader_progressbar_sftp(capsys):
    """Setup an SFTP downloader function that prints a progress bar for fetch"""
    downloader = SFTPDownloader(progressbar=True, username='demo', password='password')
    with TemporaryDirectory() as local_store:
        url = 'sftp://test.rebex.net/pub/example/pocketftp.png'
        outfile = os.path.join(local_store, 'pocketftp.png')
        downloader(url, outfile, None)
        captured = capsys.readouterr()
        printed = captured.err.split('\r')[-1].strip()
        assert len(printed) == 79
        if sys.platform == 'win32':
            progress = '100%|####################'
        else:
            progress = '100%|████████████████████'
        assert printed[:25] == progress
        assert os.path.exists(outfile)