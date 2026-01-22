from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_after_raises_exception_if_after_has_no_meaning(self):
    soup = self.soup('')
    tag = soup.new_tag('a')
    string = soup.new_string('')
    with pytest.raises(ValueError):
        string.insert_after(tag)
    with pytest.raises(NotImplementedError):
        soup.insert_after(tag)
    with pytest.raises(ValueError):
        tag.insert_after(tag)