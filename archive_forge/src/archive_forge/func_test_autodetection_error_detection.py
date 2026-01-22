import pytest
from docstring_parser.common import DocstringStyle, ParseError
from docstring_parser.parser import parse
def test_autodetection_error_detection() -> None:
    """Test autodection for the case where one of the parsers throws an error
    and another one succeeds.
    """
    source = '\n    Does something useless\n\n    :param 3 + 3 a: a param\n    '
    with pytest.raises(ParseError):
        parse(source, DocstringStyle.REST)
    docstring = parse(source)
    assert docstring
    assert docstring.style == DocstringStyle.GOOGLE