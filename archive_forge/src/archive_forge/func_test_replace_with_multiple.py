from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_with_multiple(self):
    data = '<a><b></b><c></c></a>'
    soup = self.soup(data)
    d_tag = soup.new_tag('d')
    d_tag.string = 'Text In D Tag'
    e_tag = soup.new_tag('e')
    f_tag = soup.new_tag('f')
    a_string = 'Random Text'
    soup.c.replace_with(d_tag, e_tag, a_string, f_tag)
    assert soup.decode() == '<a><b></b><d>Text In D Tag</d><e></e>Random Text<f></f></a>'
    assert soup.b.next_element == d_tag
    assert d_tag.string.next_element == e_tag
    assert e_tag.next_element.string == a_string
    assert e_tag.next_element.next_element == f_tag