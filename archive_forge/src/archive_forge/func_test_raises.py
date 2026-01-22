import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_raises() -> None:
    """Test parsing raises."""
    docstring = parse('\n        Short description\n        ')
    assert len(docstring.raises) == 0
    docstring = parse('\n        Short description\n        Raises\n        ------\n        ValueError\n            description\n        ')
    assert len(docstring.raises) == 1
    assert docstring.raises[0].type_name == 'ValueError'
    assert docstring.raises[0].description == 'description'