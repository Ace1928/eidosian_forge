from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_with_and_unwrap_give_useful_exception_when_tag_has_no_parent(self):
    soup = self.soup('<a><b>Foo</b></a><c>Bar</c>')
    a = soup.a
    a.extract()
    assert None == a.parent
    with pytest.raises(ValueError):
        a.unwrap()
    with pytest.raises(ValueError):
        a.replace_with(soup.c)