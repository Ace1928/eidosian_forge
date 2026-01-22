import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def make_structure_format(format_name, mcs, mol, subgraph, args):
    try:
        func = structure_format_functions[format_name]
    except KeyError:
        raise ValueError('Unknown format %r' % (format_name,))
    return func(mcs, mol, subgraph, args)