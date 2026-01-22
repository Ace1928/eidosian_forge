import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def parse_threshold(s):
    try:
        import fractions
    except ImportError:
        threshold = float(s)
        one = 1.0
    else:
        threshold = fractions.Fraction(s)
        one = fractions.Fraction(1)
    if not 0 <= threshold <= one:
        raise argparse.ArgumentTypeError('must be a value between 0.0 and 1.0, not %s' % s)
    return threshold