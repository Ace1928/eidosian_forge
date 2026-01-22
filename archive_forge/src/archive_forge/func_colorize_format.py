import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
def colorize_format(self, fmt, style=DEFAULT_FORMAT_STYLE):
    """
        Rewrite a logging format string to inject ANSI escape sequences.

        :param fmt: The log format string.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :returns: The logging format string with ANSI escape sequences.

        This method takes a logging format string like the ones you give to
        :class:`logging.Formatter` and processes it as follows:

        1. First the logging format string is separated into formatting
           directives versus surrounding text (according to the given `style`).

        2. Then formatting directives and surrounding text are grouped
           based on whitespace delimiters (in the surrounding text).

        3. For each group styling is selected as follows:

           1. If the group contains a single formatting directive that has
              a style defined then the whole group is styled accordingly.

           2. If the group contains multiple formatting directives that
              have styles defined then each formatting directive is styled
              individually and surrounding text isn't styled.

        As an example consider the default log format (:data:`DEFAULT_LOG_FORMAT`)::

         %(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s

        The default field styles (:data:`DEFAULT_FIELD_STYLES`) define a style for the
        `name` field but not for the `process` field, however because both fields
        are part of the same whitespace delimited token they'll be highlighted
        together in the style defined for the `name` field.
        """
    result = []
    parser = FormatStringParser(style=style)
    for group in parser.get_grouped_pairs(fmt):
        applicable_styles = [self.nn.get(self.field_styles, token.name) for token in group if token.name]
        if sum(map(bool, applicable_styles)) == 1:
            result.append(ansi_wrap(''.join((token.text for token in group)), **next((s for s in applicable_styles if s))))
        else:
            for token in group:
                text = token.text
                if token.name:
                    field_styles = self.nn.get(self.field_styles, token.name)
                    if field_styles:
                        text = ansi_wrap(text, **field_styles)
                result.append(text)
    return ''.join(result)