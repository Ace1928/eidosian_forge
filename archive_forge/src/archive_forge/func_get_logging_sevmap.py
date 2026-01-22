from __future__ import absolute_import, division, print_function
import socket
from functools import total_ordering
from itertools import count, groupby
from ansible.module_utils.six import iteritems
def get_logging_sevmap(invert=False):
    x = LOGGING_SEVMAP
    if invert:
        x = dict(map(reversed, iteritems(x)))
    return x