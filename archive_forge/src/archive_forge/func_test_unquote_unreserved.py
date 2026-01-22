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
@pytest.mark.parametrize('uri, expected', (('http://example.com/?a=%--', 'http://example.com/?a=%--'), ('http://example.com/?a=%300', 'http://example.com/?a=00')))
def test_unquote_unreserved(uri, expected):
    assert unquote_unreserved(uri) == expected