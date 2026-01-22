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
def find_username():
    """
    Find the username to include in log messages.

    :returns: A suitable username (a string).

    On UNIX systems this uses the :mod:`pwd` module which means ``root`` will
    be reported when :man:`sudo` is used (as it should). If this fails (for
    example on Windows) then :func:`getpass.getuser()` is used as a fall back.
    """
    try:
        import pwd
        uid = os.getuid()
        entry = pwd.getpwuid(uid)
        return entry.pw_name
    except Exception:
        import getpass
        return getpass.getuser()