from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_emoji_domain(self):
    """See issue #7, affecting only narrow builds (2.6-3.3)"""
    url = URL.from_text('https://xn--vi8hiv.ws')
    iri = url.to_iri()
    iri.to_text()