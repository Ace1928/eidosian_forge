from unittest.mock import patch
from docstring_parser import parse_from_object
def test_from_class_attribute_docstrings() -> None:
    """Test the parse of attribute docstrings from a class."""

    class StandardCase:
        """Short description
        Long description
        """
        attr_one: str
        'Description for attr_one'
        attr_two: bool = False
        'Description for attr_two'
    docstring = parse_from_object(StandardCase)
    assert docstring.short_description == 'Short description'
    assert docstring.long_description == 'Long description'
    assert docstring.description == 'Short description\nLong description'
    assert len(docstring.params) == 2
    assert docstring.params[0].arg_name == 'attr_one'
    assert docstring.params[0].type_name == 'str'
    assert docstring.params[0].description == 'Description for attr_one'
    assert docstring.params[1].arg_name == 'attr_two'
    assert docstring.params[1].type_name == 'bool'
    assert docstring.params[1].description == 'Description for attr_two'