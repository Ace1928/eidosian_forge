from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
@pytest.mark.skipif(not LXML_PRESENT, reason='lxml not installed, cannot parse XML document')
def test_xml_tag_inherits_self_closing_rules_from_builder(self):
    xml_soup = BeautifulSoup('', 'xml')
    xml_br = xml_soup.new_tag('br')
    xml_p = xml_soup.new_tag('p')
    assert b'<br/>' == xml_br.encode()
    assert b'<p/>' == xml_p.encode()