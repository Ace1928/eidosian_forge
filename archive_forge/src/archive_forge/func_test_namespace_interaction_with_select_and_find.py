import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
@pytest.mark.skipif(not SOUP_SIEVE_PRESENT, reason='Soup Sieve not installed')
def test_namespace_interaction_with_select_and_find(self):
    soup = self.soup('<?xml version="1.1"?>\n<root><tag xmlns="http://unprefixed-namespace.com">content</tag><prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</tag><subtag xmlns:prefix="http://another-namespace-same-prefix.com"><prefix:tag3></subtag></root>')
    assert soup.select_one('tag').name == 'tag'
    assert soup.select_one('prefix|tag2').name == 'tag2'
    assert soup.select_one('prefix|tag3') is None
    assert soup.select_one('prefix|tag3', namespaces=soup.subtag._namespaces).name == 'tag3'
    assert soup.subtag.select_one('prefix|tag3').name == 'tag3'
    assert soup.find('tag').name == 'tag'
    assert soup.find('prefix:tag2').name == 'tag2'
    assert soup.find('prefix:tag3').name == 'tag3'
    assert soup.subtag.find('prefix:tag3').name == 'tag3'