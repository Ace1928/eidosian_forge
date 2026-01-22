from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_percent_decode_mixed(self):
    assert _percent_decode('abcdé%C3%A9éfg') == 'abcdéééfg'
    assert _percent_decode('abcdé%C3éfg') == 'abcdé%C3éfg'
    with self.assertRaises(UnicodeDecodeError):
        _percent_decode('abcdé%C3éfg', raise_subencoding_exc=True)
    assert _percent_decode('é%25é', subencoding='ascii') == 'é%25é'