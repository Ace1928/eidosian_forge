import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_real_iso_8859_document(self):
    unicode_html = '<html><head><meta content="text/html; charset=ISO-8859-1" http-equiv="Content-type"/></head><body><p>Sacré bleu!</p></body></html>'
    iso_latin_html = unicode_html.encode('iso-8859-1')
    soup = self.soup(iso_latin_html)
    result = soup.encode('utf-8')
    expected = unicode_html.replace('ISO-8859-1', 'utf-8')
    expected = expected.encode('utf-8')
    assert result == expected