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
def check_style(value):
    """
    Validate a logging format style.

    :param value: The logging format style to validate (any value).
    :returns: The logging format character (a string of one character).
    :raises: :exc:`~exceptions.ValueError` when the given style isn't supported.

    On Python 3.2+ this function accepts the logging format styles ``%``, ``{``
    and ``$`` while on older versions only ``%`` is accepted (because older
    Python versions don't support alternative logging format styles).
    """
    if sys.version_info[:2] >= (3, 2):
        if value not in FORMAT_STYLE_PATTERNS:
            msg = 'Unsupported logging format style! (%r)'
            raise ValueError(format(msg, value))
    elif value != DEFAULT_FORMAT_STYLE:
        msg = 'Format string styles other than %r require Python 3.2+!'
        raise ValueError(msg, DEFAULT_FORMAT_STYLE)
    return value