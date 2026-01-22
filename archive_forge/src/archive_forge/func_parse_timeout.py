import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def parse_timeout(s):
    if s == 'none':
        return None
    timeout = float(s)
    if timeout < 0.0:
        raise argparse.ArgumentTypeError('Must be a non-negative value, not %r' % (s,))
    return timeout