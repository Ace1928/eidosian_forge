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
def find_terminal_size_using_stty():
    """
    Find the terminal size using the external command ``stty size``.

    :param stream: A stream connected to the terminal (a file object).
    :returns: A tuple of two integers with the line and column count.
    :raises: This function can raise exceptions but I'm not going to document
             them here, you should be using :func:`find_terminal_size()`.
    """
    stty = subprocess.Popen(['stty', 'size'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = stty.communicate()
    tokens = stdout.split()
    if len(tokens) != 2:
        raise Exception("Invalid output from `stty size'!")
    return tuple(map(int, tokens))