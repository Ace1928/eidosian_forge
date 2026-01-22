import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_extraction(self):
    """
        Test that extraction does not destroy the tree.

        https://bugs.launchpad.net/beautifulsoup/+bug/1782928
        """
    markup = '\n<html><head></head>\n<style>\n</style><script></script><body><p>hello</p></body></html>\n'
    soup = self.soup(markup)
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    assert len(soup.find_all('p')) == 1