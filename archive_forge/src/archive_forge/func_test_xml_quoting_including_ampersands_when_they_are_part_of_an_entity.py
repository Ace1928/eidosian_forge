import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_quoting_including_ampersands_when_they_are_part_of_an_entity(self):
    assert self.sub.substitute_xml('&Aacute;T&T') == '&amp;Aacute;T&amp;T'