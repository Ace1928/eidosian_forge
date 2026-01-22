from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_twisted_compat(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    assert durl == DecodedURL.fromText(TOTAL_URL)
    assert 'to_text' in dir(durl)
    assert 'asText' not in dir(durl)
    assert durl.to_text() == durl.asText()