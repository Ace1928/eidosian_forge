import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def get_source_and_line(self, lineno=None):
    """Return (source, line) tuple for current or given line number.

        Looks up the source and line number in the `self.input_lines`
        StringList instance to count for included source files.

        If the optional argument `lineno` is given, convert it from an
        absolute line number to the corresponding (source, line) pair.
        """
    if lineno is None:
        offset = self.line_offset
    else:
        offset = lineno - self.input_offset - 1
    try:
        src, srcoffset = self.input_lines.info(offset)
        srcline = srcoffset + 1
    except TypeError:
        src, srcline = self.get_source_and_line(offset + self.input_offset)
        return (src, srcline + 1)
    except IndexError:
        src, srcline = (None, None)
    return (src, srcline)