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
@pytest.mark.parametrize('url', ('http://192.168.1.1:5000/', 'http://192.168.1.1/', 'http://www.requests.com/'))
def test_not_bypass(self, url):
    assert get_environ_proxies(url, no_proxy=None) != {}