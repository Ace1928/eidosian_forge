import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_entities_in_attributes_converted_to_unicode(self):
    expect = '<p id="piñata"></p>'
    self.assert_soup('<p id="pi&#241;ata"></p>', expect)
    self.assert_soup('<p id="pi&#xf1;ata"></p>', expect)
    self.assert_soup('<p id="pi&#Xf1;ata"></p>', expect)
    self.assert_soup('<p id="pi&ntilde;ata"></p>', expect)