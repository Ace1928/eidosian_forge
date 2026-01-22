from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def count_looks_like(a, all_cand, comp):
    """Utility method for counting occurrences."""
    n = 0
    for b in all_cand:
        if a.info['confid'] == b.info['confid']:
            continue
        if comp.looks_like(a, b):
            n += 1
    return n