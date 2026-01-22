from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_tag_with_itself(self):
    text = '<a><b></b><c>Foo<d></d></c></a><a><e></e></a>'
    soup = self.soup(text)
    c = soup.c
    soup.c.replace_with(c)
    assert soup.decode() == self.document_for(text)