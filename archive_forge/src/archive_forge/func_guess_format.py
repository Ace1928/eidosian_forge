import errno
import os
import sys
import time
from . import archive, errors, osutils, trace
def guess_format(filename, default='dir'):
    """Guess the export format based on a file name.

    :param filename: Filename to guess from
    :param default: Default format to fall back to
    :return: format name
    """
    format = archive.format_registry.get_format_from_filename(filename)
    if format is None:
        format = default
    return format