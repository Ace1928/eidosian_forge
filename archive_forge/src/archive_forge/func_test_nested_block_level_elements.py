import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_nested_block_level_elements(self):
    """Block elements can be nested."""
    soup = self.soup('<blockquote><p><b>Foo</b></p></blockquote>')
    blockquote = soup.blockquote
    assert blockquote.p.b.string == 'Foo'
    assert blockquote.b.string == 'Foo'