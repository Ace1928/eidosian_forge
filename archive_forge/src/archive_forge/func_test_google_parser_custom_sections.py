import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_google_parser_custom_sections() -> None:
    """Test parsing an unknown section with custom GoogleParser
    configuration.
    """
    parser = GoogleParser([Section('DESCRIPTION', 'desc', SectionType.SINGULAR), Section('ARGUMENTS', 'param', SectionType.MULTIPLE), Section('ATTRIBUTES', 'attribute', SectionType.MULTIPLE), Section('EXAMPLES', 'examples', SectionType.SINGULAR)], title_colon=False)
    docstring = parser.parse('\n        DESCRIPTION\n            This is the description.\n\n        ARGUMENTS\n            arg1: first arg\n            arg2: second arg\n\n        ATTRIBUTES\n            attr1: first attribute\n            attr2: second attribute\n\n        EXAMPLES\n            Many examples\n            More examples\n        ')
    assert docstring.short_description is None
    assert docstring.long_description is None
    assert len(docstring.meta) == 6
    assert docstring.meta[0].args == ['desc']
    assert docstring.meta[0].description == 'This is the description.'
    assert docstring.meta[1].args == ['param', 'arg1']
    assert docstring.meta[1].description == 'first arg'
    assert docstring.meta[2].args == ['param', 'arg2']
    assert docstring.meta[2].description == 'second arg'
    assert docstring.meta[3].args == ['attribute', 'attr1']
    assert docstring.meta[3].description == 'first attribute'
    assert docstring.meta[4].args == ['attribute', 'attr2']
    assert docstring.meta[4].description == 'second attribute'
    assert docstring.meta[5].args == ['examples']
    assert docstring.meta[5].description == 'Many examples\nMore examples'