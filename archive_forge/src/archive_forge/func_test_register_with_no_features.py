import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_register_with_no_features(self):
    builder = self.builder_for_features()
    assert self.registry.lookup('foo') is None
    assert self.registry.lookup() == builder