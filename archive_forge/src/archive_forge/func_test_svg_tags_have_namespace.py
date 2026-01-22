import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_svg_tags_have_namespace(self):
    markup = '<svg><circle/></svg>'
    soup = self.soup(markup)
    namespace = 'http://www.w3.org/2000/svg'
    assert namespace == soup.svg.namespace
    assert namespace == soup.circle.namespace