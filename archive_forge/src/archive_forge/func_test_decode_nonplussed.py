from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_decode_nonplussed(self):
    durl = DecodedURL.from_text('/x+y%2B?a=b+c%2B', query_plus_is_space=False)
    assert durl.path == ('x+y+',)
    assert durl.get('a') == ['b+c+']
    assert durl.query == (('a', 'b+c+'),)