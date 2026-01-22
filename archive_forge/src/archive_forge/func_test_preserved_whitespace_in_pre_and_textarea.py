import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_preserved_whitespace_in_pre_and_textarea(self):
    """Whitespace must be preserved in <pre> and <textarea> tags,
        even if that would mean not prettifying the markup.
        """
    pre_markup = '<pre>a   z</pre>\n'
    textarea_markup = '<textarea> woo\nwoo  </textarea>\n'
    self.assert_soup(pre_markup)
    self.assert_soup(textarea_markup)
    soup = self.soup(pre_markup)
    assert soup.pre.prettify() == pre_markup
    soup = self.soup(textarea_markup)
    assert soup.textarea.prettify() == textarea_markup
    soup = self.soup('<textarea></textarea>')
    assert soup.textarea.prettify() == '<textarea></textarea>\n'