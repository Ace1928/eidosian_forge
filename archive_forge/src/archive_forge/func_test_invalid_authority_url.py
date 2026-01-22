from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_invalid_authority_url(self):
    self.assertRaises(URLParseError, URL.from_text, 'http://abc:\n\n/#')