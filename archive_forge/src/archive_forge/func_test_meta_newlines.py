import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
@pytest.mark.parametrize('source, expected_short_desc, expected_long_desc, expected_blank_short_desc, expected_blank_long_desc', [('\n            Short description\n            Parameters\n            ----------\n            asd\n            ', 'Short description', None, False, False), ('\n            Short description\n            Long description\n            Parameters\n            ----------\n            asd\n            ', 'Short description', 'Long description', False, False), ('\n            Short description\n            First line\n                Second line\n            Parameters\n            ----------\n            asd\n            ', 'Short description', 'First line\n    Second line', False, False), ('\n            Short description\n\n            First line\n                Second line\n            Parameters\n            ----------\n            asd\n            ', 'Short description', 'First line\n    Second line', True, False), ('\n            Short description\n\n            First line\n                Second line\n\n            Parameters\n            ----------\n            asd\n            ', 'Short description', 'First line\n    Second line', True, True), ('\n            Parameters\n            ----------\n            asd\n            ', None, None, False, False)])
def test_meta_newlines(source: str, expected_short_desc: T.Optional[str], expected_long_desc: T.Optional[str], expected_blank_short_desc: bool, expected_blank_long_desc: bool) -> None:
    """Test parsing newlines around description sections."""
    docstring = parse(source)
    assert docstring.short_description == expected_short_desc
    assert docstring.long_description == expected_long_desc
    assert docstring.blank_after_short_description == expected_blank_short_desc
    assert docstring.blank_after_long_description == expected_blank_long_desc
    assert len(docstring.meta) == 1