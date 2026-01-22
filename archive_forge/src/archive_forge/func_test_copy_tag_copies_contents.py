import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_tag_copies_contents(self):
    html = '<div><b>Foo<a></a></b><b>Bar</b></div>end'
    soup = self.soup(html)
    div = soup.div
    div_copy = copy.copy(div)
    assert str(div) == str(div_copy)
    assert div == div_copy
    assert div is not div_copy
    assert None == div_copy.parent
    assert None == div_copy.previous_element
    assert None == div_copy.find(string='Bar').next_element
    assert None != div.find(string='Bar').next_element