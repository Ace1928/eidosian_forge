import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def iter_sort_chars(self, chars):

    def upright_key(x) -> int:
        return -int(x['upright'])
    for upright_cluster in cluster_objects(list(chars), upright_key, 0):
        upright = upright_cluster[0]['upright']
        cluster_key = 'doctop' if upright else 'x0'
        subclusters = cluster_objects(upright_cluster, itemgetter(cluster_key), self.y_tolerance)
        for sc in subclusters:
            sort_key = 'x0' if upright else 'doctop'
            to_yield = sorted(sc, key=itemgetter(sort_key))
            if not (self.horizontal_ltr if upright else self.vertical_ttb):
                yield from reversed(to_yield)
            else:
                yield from to_yield