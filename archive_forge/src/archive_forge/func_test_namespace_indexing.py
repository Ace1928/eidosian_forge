import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
def test_namespace_indexing(self):
    soup = self.soup('<?xml version="1.1"?>\n<root><tag xmlns="http://unprefixed-namespace.com">content</tag><prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</prefix:tag2><prefix2:tag3 xmlns:prefix2="http://another-namespace.com"><subtag xmlns="http://another-unprefixed-namespace.com"><subsubtag xmlns="http://yet-another-unprefixed-namespace.com"></prefix2:tag3></root>')
    assert soup._namespaces == {'xml': 'http://www.w3.org/XML/1998/namespace', 'prefix': 'http://prefixed-namespace.com', 'prefix2': 'http://another-namespace.com'}
    assert soup.tag._namespaces == {'xml': 'http://www.w3.org/XML/1998/namespace'}
    assert soup.tag2._namespaces == {'prefix': 'http://prefixed-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}
    assert soup.subtag._namespaces == {'prefix2': 'http://another-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}
    assert soup.subsubtag._namespaces == {'prefix2': 'http://another-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}