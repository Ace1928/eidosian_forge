import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_strings_resembling_character_entity_references(self):
    self.assert_soup('<p>&bull; AT&T is in the s&p 500</p>', '<p>• AT&amp;T is in the s&amp;p 500</p>')