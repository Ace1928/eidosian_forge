from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_append_to_contents_moves_tag(self):
    doc = '<p id="1">Don\'t leave me <b>here</b>.</p>\n                <p id="2">Don\'t leave!</p>'
    soup = self.soup(doc)
    second_para = soup.find(id='2')
    bold = soup.b
    soup.find(id='2').append(soup.b)
    assert bold.parent == second_para
    assert soup.decode() == self.document_for('<p id="1">Don\'t leave me .</p>\n<p id="2">Don\'t leave!<b>here</b></p>')