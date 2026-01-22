import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_br_is_always_empty_element_tag(self):
    """A <br> tag is designated as an empty-element tag.

        Some parsers treat <br></br> as one <br/> tag, some parsers as
        two tags, but it should always be an empty-element tag.
        """
    soup = self.soup('<br></br>')
    assert soup.br.is_empty_element
    assert str(soup.br) == '<br/>'