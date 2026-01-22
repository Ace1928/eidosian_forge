import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_ignore_inappropriate_codecs(self):
    utf8_data = 'Räksmörgås'.encode('utf-8')
    dammit = UnicodeDammit(utf8_data, ['iso-8859-8'])
    assert dammit.original_encoding.lower() == 'utf-8'