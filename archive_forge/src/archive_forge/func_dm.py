from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def dm(d):
    result = {}
    for i, row in d.items():
        row = {j: val for j, val in row.items() if val}
        if row:
            result[i] = row
    return SDM(result, (2, 2), EXRAW)