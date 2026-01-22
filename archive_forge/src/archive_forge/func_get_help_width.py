from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def get_help_width():
    """Returns the integer width of help lines that is used in TextWrap."""
    if not sys.stdout.isatty() or termios is None or fcntl is None:
        return _DEFAULT_HELP_WIDTH
    try:
        data = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, '1234')
        columns = struct.unpack('hh', data)[1]
        if columns >= _MIN_HELP_WIDTH:
            return columns
        return int(os.getenv('COLUMNS', _DEFAULT_HELP_WIDTH))
    except (TypeError, IOError, struct.error):
        return _DEFAULT_HELP_WIDTH