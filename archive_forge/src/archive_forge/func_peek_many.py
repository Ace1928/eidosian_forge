import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def peek_many(self, number):
    self._fill_buffer(number)
    buffer = self._buffer
    if len(buffer) == number:
        ret = list(buffer)
    elif number:
        ret = []
        for t in buffer:
            ret.append(t)
            number -= 1
            if not number:
                break
    else:
        ret = []
    return ret