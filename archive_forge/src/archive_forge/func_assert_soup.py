import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def assert_soup(self, to_parse, compare_parsed_to=None):
    """Parse some markup using Beautiful Soup and verify that
        the output markup is as expected.
        """
    builder = self.default_builder
    obj = BeautifulSoup(to_parse, builder=builder)
    if compare_parsed_to is None:
        compare_parsed_to = to_parse
    assert obj.decode() == self.document_for(compare_parsed_to)
    assert all((v == 0 for v in list(obj.open_tag_counter.values())))
    assert [obj.ROOT_TAG_NAME] == [x.name for x in obj.tagStack]