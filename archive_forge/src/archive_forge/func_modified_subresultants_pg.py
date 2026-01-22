from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError
def modified_subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the ``modified'' subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    ``modified'' subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become ``modified''
    subresultants with the help of the Pell-Gordon Theorem of 1917.

    If the ``modified'' subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm sequence of the polynomials p, q.

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = (d1, d0)
        p, q = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    k = var('k')
    u_list = []
    subres_l = [p, q]
    a0, a1 = (p, q)
    del0 = d0 - d1
    degdif = del0
    rho_1 = LC(a0)
    rho_list_minus_1 = sign(LC(a0, x))
    rho1 = LC(a1, x)
    rho_list = [sign(rho1)]
    p_list = [del0]
    u = summation(k, (k, 1, p_list[0]))
    u_list.append(u)
    v = sum(p_list)
    exp_deg = d1 - 1
    a2 = -rem(a0, a1, domain=QQ)
    rho2 = LC(a2, x)
    d2 = degree(a2, x)
    deg_diff_new = exp_deg - d2
    del1 = d1 - d2
    mul_fac_old = rho1 ** (del0 + del1 - deg_diff_new)
    p_list.append(1 + deg_diff_new)
    num = 1
    for u in u_list:
        num *= (-1) ** u
    num = num * (-1) ** v
    if deg_diff_new == 0:
        den = 1
        for k in range(len(rho_list)):
            den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
    else:
        den = 1
        for k in range(len(rho_list) - 1):
            den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
        expo = p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new
        den = den * rho_list[len(rho_list) - 1] ** expo
    if sign(num / den) > 0:
        subres_l.append(simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
    else:
        subres_l.append(-simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
    k = var('k')
    rho_list.append(sign(rho2))
    u = summation(k, (k, 1, p_list[len(p_list) - 1]))
    u_list.append(u)
    v = sum(p_list)
    deg_diff_old = deg_diff_new
    while d2 > 0:
        a0, a1, d0, d1 = (a1, a2, d1, d2)
        del0 = del1
        exp_deg = d1 - 1
        a2 = -rem(a0, a1, domain=QQ)
        rho3 = LC(a2, x)
        d2 = degree(a2, x)
        deg_diff_new = exp_deg - d2
        del1 = d1 - d2
        expo_old = deg_diff_old
        expo_new = del0 + del1 - deg_diff_new
        mul_fac_new = rho2 ** expo_new * rho1 ** expo_old * mul_fac_old
        deg_diff_old, mul_fac_old = (deg_diff_new, mul_fac_new)
        rho1, rho2 = (rho2, rho3)
        p_list.append(1 + deg_diff_new)
        num = 1
        for u in u_list:
            num *= (-1) ** u
        num = num * (-1) ** v
        if deg_diff_new == 0:
            den = 1
            for k in range(len(rho_list)):
                den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
        else:
            den = 1
            for k in range(len(rho_list) - 1):
                den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
            expo = p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new
            den = den * rho_list[len(rho_list) - 1] ** expo
        if sign(num / den) > 0:
            subres_l.append(simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
        else:
            subres_l.append(-simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
        k = var('k')
        rho_list.append(sign(rho2))
        u = summation(k, (k, 1, p_list[len(p_list) - 1]))
        u_list.append(u)
        v = sum(p_list)
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)
    m = len(subres_l)
    if LC(p) < 0:
        aux_seq = [subres_l[0], subres_l[1]]
        for i in range(2, m):
            aux_seq.append(simplify(subres_l[i] * -1))
        subres_l = aux_seq
    return subres_l