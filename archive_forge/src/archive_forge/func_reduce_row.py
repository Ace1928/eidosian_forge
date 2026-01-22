from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def reduce_row(T, mu, y, rows: tuple[int, int]):
    r = closest_integer(mu[rows[0]][rows[1]])
    y[rows[0]] = [y[rows[0]][z] - r * y[rows[1]][z] for z in range(n)]
    mu[rows[0]][:rows[1]] = [mu[rows[0]][z] - r * mu[rows[1]][z] for z in range(rows[1])]
    mu[rows[0]][rows[1]] -= r
    if return_transform:
        T[rows[0]] = [T[rows[0]][z] - r * T[rows[1]][z] for z in range(m)]