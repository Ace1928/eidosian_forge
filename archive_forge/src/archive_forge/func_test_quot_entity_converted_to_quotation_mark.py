import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_quot_entity_converted_to_quotation_mark(self):
    self.assert_soup('<p>I said &quot;good day!&quot;</p>', '<p>I said "good day!"</p>')