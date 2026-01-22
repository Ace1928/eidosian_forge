import pytest
from docstring_parser.common import DocstringStyle, ParseError
from docstring_parser.parser import parse
Test autodection for the case where one of the parsers throws an error
    and another one succeeds.
    