import pytest
from docstring_parser.common import DocstringStyle, ParseError
from docstring_parser.parser import parse
def test_rest() -> None:
    """Test ReST-style parser autodetection."""
    docstring = parse('\n        Short description\n\n        Long description\n\n        Causing people to indent:\n\n            A lot sometimes\n\n        :param spam: spam desc\n        :param int bla: bla desc\n        :param str yay:\n        :raises ValueError: exc desc\n        :returns tuple: ret desc\n        ')
    assert docstring.style == DocstringStyle.REST
    assert docstring.short_description == 'Short description'
    assert docstring.long_description == 'Long description\n\nCausing people to indent:\n\n    A lot sometimes'
    assert docstring.description == 'Short description\n\nLong description\n\nCausing people to indent:\n\n    A lot sometimes'
    assert len(docstring.params) == 3
    assert docstring.params[0].arg_name == 'spam'
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == 'spam desc'
    assert docstring.params[1].arg_name == 'bla'
    assert docstring.params[1].type_name == 'int'
    assert docstring.params[1].description == 'bla desc'
    assert docstring.params[2].arg_name == 'yay'
    assert docstring.params[2].type_name == 'str'
    assert docstring.params[2].description == ''
    assert len(docstring.raises) == 1
    assert docstring.raises[0].type_name == 'ValueError'
    assert docstring.raises[0].description == 'exc desc'
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'tuple'
    assert docstring.returns.description == 'ret desc'
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns