from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_consecutive_text_nodes(self):
    soup = self.soup('<a><b>Argh!</b><c></c></a>')
    soup.b.insert(1, 'Hooray!')
    assert soup.decode() == self.document_for('<a><b>Argh!Hooray!</b><c></c></a>')
    new_text = soup.find(string='Hooray!')
    assert new_text.previous_element == 'Argh!'
    assert new_text.previous_element.next_element == new_text
    assert new_text.previous_sibling == 'Argh!'
    assert new_text.previous_sibling.next_sibling == new_text
    assert new_text.next_sibling == None
    assert new_text.next_element == soup.c