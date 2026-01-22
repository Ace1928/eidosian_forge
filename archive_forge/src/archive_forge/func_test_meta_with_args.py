import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.epydoc import compose, parse
def test_meta_with_args() -> None:
    """Test parsing meta with additional arguments."""
    docstring = parse('\n        Short description\n\n        @meta ene due rabe: asd\n        ')
    assert docstring.short_description == 'Short description'
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ['meta', 'ene', 'due', 'rabe']
    assert docstring.meta[0].description == 'asd'