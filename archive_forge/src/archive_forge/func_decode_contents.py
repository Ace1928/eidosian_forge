import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def decode_contents(self, indent_level=None, eventual_encoding=DEFAULT_OUTPUT_ENCODING, formatter='minimal'):
    """Renders the contents of this tag as a Unicode string.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The formatter decides what a
           'level' means in terms of spaces or other characters
           output.) Used internally in recursive calls while
           pretty-printing.

        :param eventual_encoding: The tag is destined to be
           encoded into this encoding. decode_contents() is _not_
           responsible for performing that encoding. This information
           is passed in so that it can be substituted in if the
           document contains a <META> tag that mentions the document's
           encoding.

        :param formatter: A Formatter object, or a string naming one of
            the standard Formatters.

        """
    return self.decode(indent_level, eventual_encoding, formatter, iterator=self.descendants)