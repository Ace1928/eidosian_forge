import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_python_specific_encodings_not_used_in_charset(self):
    for markup in [b'<meta charset="utf8"></head><meta id="encoding" charset="utf-8" />']:
        soup = self.soup(markup)
        for encoding in PYTHON_SPECIFIC_ENCODINGS:
            if encoding in ('idna', 'mbcs', 'oem', 'undefined', 'string_escape', 'string-escape'):
                continue
            encoded = soup.encode(encoding)
            assert b'meta charset=""' in encoded
            assert encoding.encode('ascii') not in encoded