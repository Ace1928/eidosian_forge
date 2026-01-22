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
@pytest.mark.parametrize('url, expected', (('http://172.16.1.1/', '172.16.1.1'), ('http://172.16.1.1:5000/', '172.16.1.1'), ('http://user:pass@172.16.1.1', '172.16.1.1'), ('http://user:pass@172.16.1.1:5000', '172.16.1.1'), ('http://hostname/', 'hostname'), ('http://hostname:5000/', 'hostname'), ('http://user:pass@hostname', 'hostname'), ('http://user:pass@hostname:5000', 'hostname')))
def test_should_bypass_proxies_pass_only_hostname(url, expected, mocker):
    """The proxy_bypass function should be called with a hostname or IP without
    a port number or auth credentials.
    """
    proxy_bypass = mocker.patch('requests.utils.proxy_bypass')
    should_bypass_proxies(url, no_proxy=None)
    proxy_bypass.assert_called_once_with(expected)