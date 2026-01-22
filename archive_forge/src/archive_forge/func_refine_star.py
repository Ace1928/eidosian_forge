import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def refine_star(self, v):
    """Refine the star domain of a vertex `v`."""
    vnn = copy.copy(v.nn)
    v1nn = []
    d_v0v1_set = set()
    for v1 in vnn:
        v1nn.append(copy.copy(v1.nn))
    for v1, v1nn in zip(vnn, v1nn):
        vnnu = v1nn.intersection(vnn)
        d_v0v1 = self.split_edge(v.x, v1.x)
        for o_d_v0v1 in d_v0v1_set:
            d_v0v1.connect(o_d_v0v1)
        d_v0v1_set.add(d_v0v1)
        for v2 in vnnu:
            d_v1v2 = self.split_edge(v1.x, v2.x)
            d_v0v1.connect(d_v1v2)
    return