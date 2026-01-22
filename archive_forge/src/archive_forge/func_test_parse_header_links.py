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
@pytest.mark.parametrize('value, expected', (('<http:/.../front.jpeg>; rel=front; type="image/jpeg"', [{'url': 'http:/.../front.jpeg', 'rel': 'front', 'type': 'image/jpeg'}]), ('<http:/.../front.jpeg>', [{'url': 'http:/.../front.jpeg'}]), ('<http:/.../front.jpeg>;', [{'url': 'http:/.../front.jpeg'}]), ('<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;', [{'url': 'http:/.../front.jpeg', 'type': 'image/jpeg'}, {'url': 'http://.../back.jpeg'}]), ('', [])))
def test_parse_header_links(value, expected):
    assert parse_header_links(value) == expected