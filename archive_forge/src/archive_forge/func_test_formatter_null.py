import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_null(self):
    markup = '<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
    soup = self.soup(markup)
    decoded = soup.decode(formatter=None)
    assert decoded == self.document_for('<b><<Sacré bleu!>></b>')