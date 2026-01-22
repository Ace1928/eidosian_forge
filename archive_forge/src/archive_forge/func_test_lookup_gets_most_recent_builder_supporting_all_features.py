import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_gets_most_recent_builder_supporting_all_features(self):
    has_one = self.builder_for_features('foo')
    has_the_other = self.builder_for_features('bar')
    has_both_early = self.builder_for_features('foo', 'bar', 'baz')
    has_both_late = self.builder_for_features('foo', 'bar', 'quux')
    lacks_one = self.builder_for_features('bar')
    has_the_other = self.builder_for_features('foo')
    assert self.registry.lookup('foo', 'bar') == has_both_late
    assert self.registry.lookup('foo', 'bar', 'baz') == has_both_early