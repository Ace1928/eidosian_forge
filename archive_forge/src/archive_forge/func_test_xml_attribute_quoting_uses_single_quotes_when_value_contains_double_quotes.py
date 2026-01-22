import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_attribute_quoting_uses_single_quotes_when_value_contains_double_quotes(self):
    s = 'Welcome to "my bar"'
    assert self.sub.substitute_xml(s, True) == '\'Welcome to "my bar"\''