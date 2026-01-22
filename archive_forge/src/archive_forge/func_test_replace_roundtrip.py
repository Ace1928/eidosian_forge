from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_replace_roundtrip(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    durl2 = durl.replace(scheme=durl.scheme, host=durl.host, path=durl.path, query=durl.query, fragment=durl.fragment, port=durl.port, rooted=durl.rooted, userinfo=durl.userinfo, uses_netloc=durl.uses_netloc)
    assert durl == durl2