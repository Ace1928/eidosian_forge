import asyncio
import os
from importlib import import_module
from pathlib import Path
from posixpath import split
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type
from unittest import TestCase, mock
from twisted.internet.defer import Deferred
from twisted.trial.unittest import SkipTest
from scrapy import Spider
from scrapy.crawler import Crawler
from scrapy.utils.boto import is_botocore_available
def get_ftp_content_and_delete(path: str, host: str, port: int, username: str, password: str, use_active_mode: bool=False) -> bytes:
    from ftplib import FTP
    ftp = FTP()
    ftp.connect(host, port)
    ftp.login(username, password)
    if use_active_mode:
        ftp.set_pasv(False)
    ftp_data: List[bytes] = []

    def buffer_data(data: bytes) -> None:
        ftp_data.append(data)
    ftp.retrbinary(f'RETR {path}', buffer_data)
    dirname, filename = split(path)
    ftp.cwd(dirname)
    ftp.delete(filename)
    return b''.join(ftp_data)