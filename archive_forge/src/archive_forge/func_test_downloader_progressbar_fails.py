import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.skipif(tqdm is not None, reason='tqdm must be missing')
@pytest.mark.parametrize('downloader', [HTTPDownloader, FTPDownloader, SFTPDownloader])
def test_downloader_progressbar_fails(downloader):
    """Make sure an error is raised if trying to use progressbar without tqdm"""
    with pytest.raises(ValueError) as exc:
        downloader(progressbar=True)
    assert "'tqdm'" in str(exc.value)