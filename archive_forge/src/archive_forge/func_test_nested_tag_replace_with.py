from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_nested_tag_replace_with(self):
    soup = self.soup('<a>We<b>reserve<c>the</c><d>right</d></b></a><e>to<f>refuse</f><g>service</g></e>')
    remove_tag = soup.b
    move_tag = soup.f
    remove_tag.replace_with(move_tag)
    assert soup.decode() == self.document_for('<a>We<f>refuse</f></a><e>to<g>service</g></e>')
    assert remove_tag.parent == None
    assert remove_tag.find(string='right').next_element == None
    assert remove_tag.previous_element == None
    assert remove_tag.next_sibling == None
    assert remove_tag.previous_sibling == None
    assert move_tag.parent == soup.a
    assert move_tag.previous_element == 'We'
    assert move_tag.next_element.next_element == soup.e
    assert move_tag.next_sibling == None
    to_text = soup.find(string='to')
    g_tag = soup.g
    assert to_text.next_element == g_tag
    assert to_text.next_sibling == g_tag
    assert g_tag.previous_element == to_text
    assert g_tag.previous_sibling == to_text