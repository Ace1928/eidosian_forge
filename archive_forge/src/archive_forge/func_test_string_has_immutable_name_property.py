import pytest
from bs4.element import (
from . import SoupTest
def test_string_has_immutable_name_property(self):
    string = self.soup('s').string
    assert None == string.name
    with pytest.raises(AttributeError):
        string.name = 'foo'