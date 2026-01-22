import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_attribute_values_with_nested_quotes_are_left_alone(self):
    text = '<foo attr=\'bar "brawls" happen\'>a</foo>'
    self.assert_soup(text)