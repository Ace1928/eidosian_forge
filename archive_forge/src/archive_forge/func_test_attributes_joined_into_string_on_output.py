import warnings
from bs4.element import (
from . import SoupTest
def test_attributes_joined_into_string_on_output(self):
    soup = self.soup("<a class='foo\tbar'>")
    assert b'<a class="foo bar"></a>' == soup.a.encode()