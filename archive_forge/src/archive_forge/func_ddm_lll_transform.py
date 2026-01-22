from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def ddm_lll_transform(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=True)