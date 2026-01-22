import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def takewhile(self, predicate):
    """Variant of itertools.takewhile except it does not discard the first non-matching token"""
    buffer = self._buffer
    while buffer or self._fill_buffer(5):
        v = buffer[0]
        if predicate(v):
            buffer.popleft()
            yield v
        else:
            break