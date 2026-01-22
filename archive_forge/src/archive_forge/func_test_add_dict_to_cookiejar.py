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
@pytest.mark.parametrize('cookiejar', (compat.cookielib.CookieJar(), RequestsCookieJar()))
def test_add_dict_to_cookiejar(cookiejar):
    """Ensure add_dict_to_cookiejar works for
    non-RequestsCookieJar CookieJars
    """
    cookiedict = {'test': 'cookies', 'good': 'cookies'}
    cj = add_dict_to_cookiejar(cookiejar, cookiedict)
    cookies = {cookie.name: cookie.value for cookie in cj}
    assert cookiedict == cookies