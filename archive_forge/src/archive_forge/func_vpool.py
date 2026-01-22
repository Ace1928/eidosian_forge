import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def vpool(self, origin, supremum):
    vot = tuple(origin)
    vst = tuple(supremum)
    vo = self.V[vot]
    vs = self.V[vst]
    bl = list(vot)
    bu = list(vst)
    for i, (voi, vsi) in enumerate(zip(vot, vst)):
        if bl[i] > vsi:
            bl[i] = vsi
        if bu[i] < voi:
            bu[i] = voi
    vn_pool = set()
    vn_pool.update(vo.nn)
    vn_pool.update(vs.nn)
    cvn_pool = copy.copy(vn_pool)
    for vn in cvn_pool:
        for i, xi in enumerate(vn.x):
            if bl[i] <= xi <= bu[i]:
                pass
            else:
                try:
                    vn_pool.remove(vn)
                except KeyError:
                    pass
    return vn_pool