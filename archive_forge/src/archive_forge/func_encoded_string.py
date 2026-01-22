from __future__ import absolute_import
import re
import sys
def encoded_string(s, encoding):
    assert isinstance(s, (_unicode, bytes))
    s = EncodedString(s)
    if encoding is not None:
        s.encoding = encoding
    return s