import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_attribute_quoting_normally_uses_double_quotes(self):
    assert self.sub.substitute_xml('Welcome', True) == '"Welcome"'
    assert self.sub.substitute_xml("Bob's Bar", True) == '"Bob\'s Bar"'