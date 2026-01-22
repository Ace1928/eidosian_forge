import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_default_formatter_is_minimal(self):
    markup = '<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
    soup = self.soup(markup)
    decoded = soup.decode(formatter='minimal')
    assert decoded == self.document_for('<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>')