from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_passthroughs(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    assert durl.sibling('te%t').path[-1] == 'te%t'
    assert durl.child('../test2%').path[-1] == '../test2%'
    assert durl.child() == durl
    assert durl.child() is durl
    assert durl.click('/').path[-1] == ''
    assert durl.user == 'user'
    assert '.' in durl.path
    assert '.' not in durl.normalize().path
    assert durl.to_uri().fragment == 'fr%C3%A9g'
    assert ' ' in durl.to_iri().path[1]
    assert durl.to_text(with_password=True) == TOTAL_URL
    assert durl.absolute
    assert durl.rooted
    assert durl == durl.encoded_url.get_decoded_url()
    durl2 = DecodedURL.from_text(TOTAL_URL, lazy=True)
    assert durl2 == durl2.encoded_url.get_decoded_url(lazy=True)
    assert str(DecodedURL.from_text(BASIC_URL).child(' ')) == 'http://example.com/%20'
    assert not durl == 1
    assert durl != 1