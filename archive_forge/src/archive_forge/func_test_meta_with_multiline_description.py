import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_meta_with_multiline_description() -> None:
    """Test parsing multiline meta documentation."""
    docstring = parse('\n        Short description\n\n        Parameters\n        ----------\n        spam\n            asd\n            1\n                2\n            3\n        ')
    assert docstring.short_description == 'Short description'
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ['param', 'spam']
    assert docstring.meta[0].arg_name == 'spam'
    assert docstring.meta[0].description == 'asd\n1\n    2\n3'