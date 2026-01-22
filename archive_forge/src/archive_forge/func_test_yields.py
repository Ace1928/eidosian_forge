import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_yields() -> None:
    """Test parsing yields."""
    docstring = parse('\n        Short description\n        Yields\n        ------\n        int\n            description\n        ')
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ['yields']
    assert docstring.meta[0].type_name == 'int'
    assert docstring.meta[0].description == 'description'
    assert docstring.meta[0].return_name is None
    assert docstring.meta[0].is_generator