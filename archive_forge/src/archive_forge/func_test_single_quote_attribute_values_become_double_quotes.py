import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_single_quote_attribute_values_become_double_quotes(self):
    self.assert_soup("<foo attr='bar'></foo>", '<foo attr="bar"></foo>')