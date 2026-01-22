from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
def get_sync_start_position(self, document, lineno):
    """ Scan backwards, and find a possible position to start. """
    pattern = self._compiled_pattern
    lines = document.lines
    for i in range(lineno, max(-1, lineno - self.MAX_BACKWARDS), -1):
        match = pattern.match(lines[i])
        if match:
            return (i, match.start())
    if lineno < self.FROM_START_IF_NO_SYNC_POS_FOUND:
        return (0, 0)
    else:
        return (lineno, 0)