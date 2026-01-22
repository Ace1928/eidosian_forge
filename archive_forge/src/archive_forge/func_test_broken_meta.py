import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.epydoc import compose, parse
def test_broken_meta() -> None:
    """Test parsing broken meta."""
    with pytest.raises(ParseError):
        parse('@')
    with pytest.raises(ParseError):
        parse('@param herp derp')
    with pytest.raises(ParseError):
        parse('@param: invalid')
    with pytest.raises(ParseError):
        parse('@param with too many args: desc')
    parse('@sthstrange: desc')