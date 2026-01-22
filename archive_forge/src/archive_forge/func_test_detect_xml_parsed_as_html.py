import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_detect_xml_parsed_as_html(self):
    markup = b'<?xml version="1.0" encoding="utf-8"?><tag>string</tag>'
    with warnings.catch_warnings(record=True) as w:
        soup = self.soup(markup)
        assert soup.tag.string == 'string'
    [warning] = w
    assert isinstance(warning.message, XMLParsedAsHTMLWarning)
    assert str(warning.message) == XMLParsedAsHTMLWarning.MESSAGE