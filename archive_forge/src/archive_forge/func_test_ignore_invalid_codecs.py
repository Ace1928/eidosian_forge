import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_ignore_invalid_codecs(self):
    utf8_data = 'Räksmörgås'.encode('utf-8')
    for bad_encoding in ['.utf8', '...', 'utF---16.!']:
        dammit = UnicodeDammit(utf8_data, [bad_encoding])
        assert dammit.original_encoding.lower() == 'utf-8'