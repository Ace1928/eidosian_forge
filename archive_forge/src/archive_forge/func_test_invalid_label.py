from __future__ import unicode_literals
from . import (lookup, LABELS, decode, encode, iter_decode, iter_encode,
def test_invalid_label():
    assert_raises(LookupError, decode, b'\xef\xbb\xbf\xc3\xa9', 'invalid')
    assert_raises(LookupError, encode, 'é', 'invalid')
    assert_raises(LookupError, iter_decode, [], 'invalid')
    assert_raises(LookupError, iter_encode, [], 'invalid')
    assert_raises(LookupError, IncrementalDecoder, 'invalid')
    assert_raises(LookupError, IncrementalEncoder, 'invalid')