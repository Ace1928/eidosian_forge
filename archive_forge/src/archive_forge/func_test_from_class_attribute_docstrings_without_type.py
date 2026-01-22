from unittest.mock import patch
from docstring_parser import parse_from_object
def test_from_class_attribute_docstrings_without_type() -> None:
    """Test the parse of untyped attribute docstrings."""

    class WithoutType:
        attr_one = 'value'
        'Description for attr_one'
    docstring = parse_from_object(WithoutType)
    assert docstring.short_description is None
    assert docstring.long_description is None
    assert docstring.description is None
    assert len(docstring.params) == 1
    assert docstring.params[0].arg_name == 'attr_one'
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == 'Description for attr_one'