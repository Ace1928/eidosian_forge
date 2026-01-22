import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_attribute_quoting_escapes_single_quotes_when_value_contains_both_single_and_double_quotes(self):
    s = 'Welcome to "Bob\'s Bar"'
    assert self.sub.substitute_xml(s, True) == '"Welcome to &quot;Bob\'s Bar&quot;"'