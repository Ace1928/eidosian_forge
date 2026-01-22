import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
@pytest.mark.parametrize('multi_valued_attributes', [None, {}, dict(b=['class']), {'*': ['notclass']}])
def test_attribute_not_multi_valued(self, multi_valued_attributes):
    markup = '<html xmlns="http://www.w3.org/1999/xhtml"><a class="a b c"></html>'
    soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
    assert soup.a['class'] == 'a b c'