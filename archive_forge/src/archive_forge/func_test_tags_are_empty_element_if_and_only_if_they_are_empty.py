import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_tags_are_empty_element_if_and_only_if_they_are_empty(self):
    self.assert_soup('<p>', '<p/>')
    self.assert_soup('<p>foo</p>')