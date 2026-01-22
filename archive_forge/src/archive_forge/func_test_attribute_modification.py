from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_attribute_modification(self):
    soup = self.soup('<a id="1"></a>')
    soup.a['id'] = 2
    assert soup.decode() == self.document_for('<a id="2"></a>')
    del soup.a['id']
    assert soup.decode() == self.document_for('<a></a>')
    soup.a['id2'] = 'foo'
    assert soup.decode() == self.document_for('<a id2="foo"></a>')