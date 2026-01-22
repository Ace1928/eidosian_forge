from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def invariant_factors(m):
    """
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    """
    domain = m.domain
    if not domain.is_PID:
        msg = 'The matrix entries must be over a principal ideal domain'
        raise ValueError(msg)
    if 0 in m.shape:
        return ()
    rows, cols = shape = m.shape
    m = list(m.to_dense().rep)

    def add_rows(m, i, j, a, b, c, d):
        for k in range(cols):
            e = m[i][k]
            m[i][k] = a * e + b * m[j][k]
            m[j][k] = c * e + d * m[j][k]

    def clear_column(m):
        if m[0][0] == 0:
            return m
        pivot = m[0][0]
        for j in range(1, rows):
            if m[j][0] == 0:
                continue
            d, r = domain.div(m[j][0], pivot)
            if r == 0:
                add_rows(m, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])
                d_0 = domain.div(m[j][0], g)[0]
                d_j = domain.div(pivot, g)[0]
                add_rows(m, 0, j, a, b, d_0, -d_j)
                pivot = g
        return m

    def clear_row(m):
        if m[0][0] == 0:
            return m
        pivot = m[0][0]
        for j in range(1, cols):
            if m[0][j] == 0:
                continue
            d, r = domain.div(m[0][j], pivot)
            if r == 0:
                add_columns(m, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])
                d_0 = domain.div(m[0][j], g)[0]
                d_j = domain.div(pivot, g)[0]
                add_columns(m, 0, j, a, b, d_0, -d_j)
                pivot = g
        return m
    ind = [i for i in range(rows) if m[i][0] != 0]
    if ind and ind[0] != 0:
        m[0], m[ind[0]] = (m[ind[0]], m[0])
    else:
        ind = [j for j in range(cols) if m[0][j] != 0]
        if ind and ind[0] != 0:
            for row in m:
                row[0], row[ind[0]] = (row[ind[0]], row[0])
    while any((m[0][i] != 0 for i in range(1, cols))) or any((m[i][0] != 0 for i in range(1, rows))):
        m = clear_column(m)
        m = clear_row(m)
    if 1 in shape:
        invs = ()
    else:
        lower_right = DomainMatrix([r[1:] for r in m[1:]], (rows - 1, cols - 1), domain)
        invs = invariant_factors(lower_right)
    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        for i in range(len(result) - 1):
            if result[i] and domain.div(result[i + 1], result[i])[1] != 0:
                g = domain.gcd(result[i + 1], result[i])
                result[i + 1] = domain.div(result[i], g)[0] * result[i + 1]
                result[i] = g
            else:
                break
    else:
        result = invs + (m[0][0],)
    return tuple(result)