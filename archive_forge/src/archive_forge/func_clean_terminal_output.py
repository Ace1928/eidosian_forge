import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
def clean_terminal_output(text):
    """
    Clean up the terminal output of a command.

    :param text: The raw text with special characters (a Unicode string).
    :returns: A list of Unicode strings (one for each line).

    This function emulates the effect of backspace (0x08), carriage return
    (0x0D) and line feed (0x0A) characters and the ANSI 'erase line' escape
    sequence on interactive terminals. It's intended to clean up command output
    that was originally meant to be rendered on an interactive terminal and
    that has been captured using e.g. the :man:`script` program [#]_ or the
    :mod:`pty` module [#]_.

    .. [#] My coloredlogs_ package supports the ``coloredlogs --to-html``
           command which uses :man:`script` to fool a subprocess into thinking
           that it's connected to an interactive terminal (in order to get it
           to emit ANSI escape sequences).

    .. [#] My capturer_ package uses the :mod:`pty` module to fool the current
           process and subprocesses into thinking they are connected to an
           interactive terminal (in order to get them to emit ANSI escape
           sequences).

    **Some caveats about the use of this function:**

    - Strictly speaking the effect of carriage returns cannot be emulated
      outside of an actual terminal due to the interaction between overlapping
      output, terminal widths and line wrapping. The goal of this function is
      to sanitize noise in terminal output while preserving useful output.
      Think of it as a useful and pragmatic but possibly lossy conversion.

    - The algorithm isn't smart enough to properly handle a pair of ANSI escape
      sequences that open before a carriage return and close after the last
      carriage return in a linefeed delimited string; the resulting string will
      contain only the closing end of the ANSI escape sequence pair. Tracking
      this kind of complexity requires a state machine and proper parsing.

    .. _capturer: https://pypi.org/project/capturer
    .. _coloredlogs: https://pypi.org/project/coloredlogs
    """
    cleaned_lines = []
    current_line = ''
    current_position = 0
    for token in CLEAN_OUTPUT_PATTERN.split(text):
        if token == '\r':
            current_position = 0
        elif token == '\x08':
            current_position = max(0, current_position - 1)
        else:
            if token == '\n':
                cleaned_lines.append(current_line)
            if token in ('\n', ANSI_ERASE_LINE):
                current_line = ''
                current_position = 0
            elif token:
                new_position = current_position + len(token)
                prefix = current_line[:current_position]
                suffix = current_line[new_position:]
                current_line = prefix + token + suffix
                current_position = new_position
    cleaned_lines.append(current_line)
    while cleaned_lines and (not cleaned_lines[-1]):
        cleaned_lines.pop(-1)
    return cleaned_lines