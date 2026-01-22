import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_real_shift_jis_document(self):
    shift_jis_html = b'<html><head></head><body><pre>\x82\xb1\x82\xea\x82\xcdShift-JIS\x82\xc5\x83R\x81[\x83f\x83B\x83\x93\x83O\x82\xb3\x82\xea\x82\xbd\x93\xfa\x96{\x8c\xea\x82\xcc\x83t\x83@\x83C\x83\x8b\x82\xc5\x82\xb7\x81B</pre></body></html>'
    unicode_html = shift_jis_html.decode('shift-jis')
    soup = self.soup(unicode_html)
    assert soup.encode('utf-8') == unicode_html.encode('utf-8')
    assert soup.encode('euc_jp') == unicode_html.encode('euc_jp')