from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def read_until(self, terminator):
    """Read the input stream until the terminator is found.

        Throws MissingTerminator if the terminator is not found.

        Note: This method does not read from the line buffer.

        :return: the bytes read up to but excluding the terminator.
        """
    lines = []
    term = terminator + b'\n'
    while True:
        line = self.input.readline()
        if line == term:
            break
        else:
            lines.append(line)
    return b''.join(lines)