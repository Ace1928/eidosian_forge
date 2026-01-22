from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def extract_slice(self, slice1, slice2):
    m, n = self.shape
    ri = range(m)[slice1]
    ci = range(n)[slice2]
    sdm = {}
    for i, row in self.items():
        if i in ri:
            row = {ci.index(j): e for j, e in row.items() if j in ci}
            if row:
                sdm[ri.index(i)] = row
    return self.new(sdm, (len(ri), len(ci)), self.domain)