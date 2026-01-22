import os
import copy
import filecmp
from io import BytesIO
import tarfile
import zipfile
from collections import deque
import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
from requests._internal_utils import unicode_is_ascii
from .compat import StringIO, cStringIO
@pytest.mark.parametrize('url', ('http://192.168.0.1:5000/', 'http://192.168.0.1/', 'http://172.16.1.1/', 'http://172.16.1.1:5000/', 'http://localhost.localdomain:5000/v1.0/'))
def test_not_bypass_no_proxy_keyword(self, url, monkeypatch):
    monkeypatch.setenv('http_proxy', 'http://proxy.example.com:3128/')
    no_proxy = '192.168.1.1,requests.com'
    assert get_environ_proxies(url, no_proxy=no_proxy) != {}