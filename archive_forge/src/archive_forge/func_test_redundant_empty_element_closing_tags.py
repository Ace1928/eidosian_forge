from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def test_redundant_empty_element_closing_tags(self):
    self.assert_soup('<br></br><br></br><br></br>', '<br/><br/><br/>')
    self.assert_soup('</br></br></br>', '')