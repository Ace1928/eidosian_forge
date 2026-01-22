import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_find_declared_encoding(self):
    html_unicode = '<html><head><meta charset="utf-8"></head></html>'
    html_bytes = html_unicode.encode('ascii')
    xml_unicode = '<?xml version="1.0" encoding="ISO-8859-1" ?>'
    xml_bytes = xml_unicode.encode('ascii')
    m = EncodingDetector.find_declared_encoding
    assert m(html_unicode, is_html=False) is None
    assert 'utf-8' == m(html_unicode, is_html=True)
    assert 'utf-8' == m(html_bytes, is_html=True)
    assert 'iso-8859-1' == m(xml_unicode)
    assert 'iso-8859-1' == m(xml_bytes)
    spacer = b' ' * 5000
    assert m(spacer + html_bytes) is None
    assert m(spacer + xml_bytes) is None
    assert m(spacer + html_bytes, is_html=True, search_entire_document=True) == 'utf-8'
    assert m(xml_bytes, search_entire_document=True) == 'iso-8859-1'
    assert m(b' ' + xml_bytes, search_entire_document=True) == 'iso-8859-1'
    assert m(b'a' + xml_bytes, search_entire_document=True) is None