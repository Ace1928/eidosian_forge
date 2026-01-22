import re
import sys
import typing
from .util import (
from .unicode import pyparsing_unicode as ppu
def mark_input_line(self, marker_string: typing.Optional[str]=None, *, markerString: str='>!<') -> str:
    """
        Extracts the exception line from the input string, and marks
        the location of the exception with a special symbol.
        """
    markerString = marker_string if marker_string is not None else markerString
    line_str = self.line
    line_column = self.column - 1
    if markerString:
        line_str = ''.join((line_str[:line_column], markerString, line_str[line_column:]))
    return line_str.strip()