import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.skipif(paramiko is not None, reason='paramiko must be missing')
def test_sftp_downloader_fail_if_paramiko_missing():
    """test must fail if paramiko is not installed"""
    with pytest.raises(ValueError) as exc:
        SFTPDownloader()
    assert "'paramiko'" in str(exc.value)