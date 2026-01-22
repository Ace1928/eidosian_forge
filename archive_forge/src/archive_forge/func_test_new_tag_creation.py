from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_new_tag_creation(self):
    builder = builder_registry.lookup('html')()
    soup = self.soup('<body></body>', builder=builder)
    a = Tag(soup, builder, 'a')
    ol = Tag(soup, builder, 'ol')
    a['href'] = 'http://foo.com/'
    soup.body.insert(0, a)
    soup.body.insert(1, ol)
    assert soup.body.encode() == b'<body><a href="http://foo.com/"></a><ol></ol></body>'