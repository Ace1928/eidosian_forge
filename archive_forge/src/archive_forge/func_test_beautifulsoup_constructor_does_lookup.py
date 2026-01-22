import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_beautifulsoup_constructor_does_lookup(self):
    with warnings.catch_warnings(record=True) as w:
        BeautifulSoup('', features='html')
        BeautifulSoup('', features=['html', 'fast'])
        pass
    with pytest.raises(ValueError):
        BeautifulSoup('', features='no-such-feature')