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
@pytest.mark.parametrize('value, expected', (('foo="is a fish", bar="as well"', {'foo': 'is a fish', 'bar': 'as well'}), ('key_without_value', {'key_without_value': None})))
def test_parse_dict_header(value, expected):
    assert parse_dict_header(value) == expected