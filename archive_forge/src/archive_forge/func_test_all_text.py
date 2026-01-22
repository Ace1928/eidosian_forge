import warnings
from bs4.element import (
from . import SoupTest
def test_all_text(self):
    """Tag.text and Tag.get_text(sep=u"") -> all child text, concatenated"""
    soup = self.soup('<a>a<b>r</b>   <r> t </r></a>')
    assert soup.a.text == 'ar  t '
    assert soup.a.get_text(strip=True) == 'art'
    assert soup.a.get_text(',') == 'a,r, , t '
    assert soup.a.get_text(',', strip=True) == 'a,r,t'