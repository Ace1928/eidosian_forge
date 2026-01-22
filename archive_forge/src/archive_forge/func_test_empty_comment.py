import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_empty_comment(self):
    """
        Test that empty comment does not break structure.

        https://bugs.launchpad.net/beautifulsoup/+bug/1806598
        """
    markup = '\n<html>\n<body>\n<form>\n<!----><input type="text">\n</form>\n</body>\n</html>\n'
    soup = self.soup(markup)
    inputs = []
    for form in soup.find_all('form'):
        inputs.extend(form.find_all('input'))
    assert len(inputs) == 1