from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_wrong_constructor(self):
    with self.assertRaises(ValueError):
        URL(BASIC_URL)
    with self.assertRaises(ValueError):
        URL('HTTP_____more_like_imHoTTeP')