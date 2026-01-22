import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
@pytest.mark.parametrize('original,substituted', [('foo∀☃õbar', 'foo&forall;☃&otilde;bar'), ('‘’foo“”', '&lsquo;&rsquo;foo&ldquo;&rdquo;')])
def test_substitute_html(self, original, substituted):
    assert self.sub.substitute_html(original) == substituted