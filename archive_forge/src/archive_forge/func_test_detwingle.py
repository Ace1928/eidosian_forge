import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_detwingle(self):
    utf8 = ('☃' * 3).encode('utf8')
    windows_1252 = '“Hi, I like Windows!”'.encode('windows_1252')
    doc = utf8 + windows_1252 + utf8
    with pytest.raises(UnicodeDecodeError):
        doc.decode('utf8')
    fixed = UnicodeDammit.detwingle(doc)
    assert '☃☃☃“Hi, I like Windows!”☃☃☃' == fixed.decode('utf8')