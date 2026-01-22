import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def gen_refs_recursive(ref, nobj, left, total, depth):
    points = []
    if depth == nobj - 1:
        ref[depth] = left / total
        points.append(ref)
    else:
        for i in range(left + 1):
            ref[depth] = i / total
            points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
    return points