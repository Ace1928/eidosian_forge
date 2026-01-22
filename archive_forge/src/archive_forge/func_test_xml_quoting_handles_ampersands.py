import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_quoting_handles_ampersands(self):
    assert self.sub.substitute_xml('AT&T') == 'AT&amp;T'