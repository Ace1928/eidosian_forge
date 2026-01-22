import pytest
from bs4.element import Tag
from bs4.formatter import (
from . import SoupTest
def test_default_indent_value(self):
    formatter = Formatter()
    assert formatter.indent == ' '