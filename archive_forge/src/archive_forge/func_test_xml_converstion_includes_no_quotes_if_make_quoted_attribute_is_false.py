import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_converstion_includes_no_quotes_if_make_quoted_attribute_is_false(self):
    s = 'Welcome to "my bar"'
    assert self.sub.substitute_xml(s, False) == s