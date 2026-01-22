import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_fails_when_no_builder_implements_feature(self):
    builder = self.builder_for_features('foo', 'bar')
    assert self.registry.lookup('baz') is None