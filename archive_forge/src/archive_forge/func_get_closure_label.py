import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_closure_label(bond_smarts, closure):
    if closure < 10:
        return bond_smarts + str(closure)
    return bond_smarts + f'%{closure:02d}'