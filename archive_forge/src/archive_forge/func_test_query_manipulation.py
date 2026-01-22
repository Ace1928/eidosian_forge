from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_query_manipulation(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    assert durl.get('zot') == ['23%']
    durl = durl.add(' ', 'space')
    assert durl.get(' ') == ['space']
    durl = durl.set(' ', 'spa%ed')
    assert durl.get(' ') == ['spa%ed']
    durl = DecodedURL(url=durl.to_uri())
    assert durl.get(' ') == ['spa%ed']
    durl = durl.remove(' ')
    assert durl.get(' ') == []
    durl = DecodedURL.from_text('/?%61rg=b&arg=c')
    assert durl.get('arg') == ['b', 'c']
    assert durl.set('arg', 'd').get('arg') == ['d']
    durl = DecodedURL.from_text('https://example.com/a/b/?fóó=1&bar=2&fóó=3')
    assert durl.remove('fóó') == DecodedURL.from_text('https://example.com/a/b/?bar=2')
    assert durl.remove('fóó', value='1') == DecodedURL.from_text('https://example.com/a/b/?bar=2&fóó=3')
    assert durl.remove('fóó', limit=1) == DecodedURL.from_text('https://example.com/a/b/?bar=2&fóó=3')
    assert durl.remove('fóó', value='1', limit=0) == DecodedURL.from_text('https://example.com/a/b/?fóó=1&bar=2&fóó=3')