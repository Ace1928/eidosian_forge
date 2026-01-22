from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_isolate_complex_roots_sqf(f, K, eps=None, inf=None, sup=None, blackbox=False):
    """Isolate complex roots of a square-free polynomial using Collins-Krandick algorithm. """
    if not K.is_ZZ and (not K.is_QQ):
        raise DomainError('isolation of complex roots is not supported over %s' % K)
    if dup_degree(f) <= 0:
        return []
    if K.is_ZZ:
        F = K.get_field()
    else:
        F = K
    f = dup_convert(f, K, F)
    lc = abs(dup_LC(f, F))
    B = 2 * max([F.quo(abs(c), lc) for c in f])
    (u, v), (s, t) = ((-B, F.zero), (B, B))
    if inf is not None:
        u = inf
    if sup is not None:
        s = sup
    if v < 0 or t <= v or s <= u:
        raise ValueError('not a valid complex isolation rectangle')
    f1, f2 = dup_real_imag(f, F)
    f1L1 = dmp_eval_in(f1, v, 1, 1, F)
    f2L1 = dmp_eval_in(f2, v, 1, 1, F)
    f1L2 = dmp_eval_in(f1, s, 0, 1, F)
    f2L2 = dmp_eval_in(f2, s, 0, 1, F)
    f1L3 = dmp_eval_in(f1, t, 1, 1, F)
    f2L3 = dmp_eval_in(f2, t, 1, 1, F)
    f1L4 = dmp_eval_in(f1, u, 0, 1, F)
    f2L4 = dmp_eval_in(f2, u, 0, 1, F)
    S_L1 = [f1L1, f2L1]
    S_L2 = [f1L2, f2L2]
    S_L3 = [f1L3, f2L3]
    S_L4 = [f1L4, f2L4]
    I_L1 = dup_isolate_real_roots_list(S_L1, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)
    Q_L1 = _intervals_to_quadrants(I_L1, f1L1, f2L1, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2, f2L2, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3, f2L3, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4, f2L4, t, v, F)
    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4)
    N = _winding_number(T, F)
    if not N:
        return []
    I = (I_L1, I_L2, I_L3, I_L4)
    Q = (Q_L1, Q_L2, Q_L3, Q_L4)
    F1 = (f1L1, f1L2, f1L3, f1L4)
    F2 = (f2L1, f2L2, f2L3, f2L4)
    rectangles, roots = ([(N, (u, v), (s, t), I, Q, F1, F2)], [])
    while rectangles:
        N, (u, v), (s, t), I, Q, F1, F2 = _depth_first_select(rectangles)
        if s - u > t - v:
            D_L, D_R = _vertical_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)
            N_L, a, b, I_L, Q_L, F1_L, F2_L = D_L
            N_R, c, d, I_R, Q_R, F1_R, F2_R = D_R
            if N_L >= 1:
                if N_L == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(a, b, I_L, Q_L, F1_L, F2_L, f1, f2, F))
                else:
                    rectangles.append(D_L)
            if N_R >= 1:
                if N_R == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(c, d, I_R, Q_R, F1_R, F2_R, f1, f2, F))
                else:
                    rectangles.append(D_R)
        else:
            D_B, D_U = _horizontal_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)
            N_B, a, b, I_B, Q_B, F1_B, F2_B = D_B
            N_U, c, d, I_U, Q_U, F1_U, F2_U = D_U
            if N_B >= 1:
                if N_B == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(a, b, I_B, Q_B, F1_B, F2_B, f1, f2, F))
                else:
                    rectangles.append(D_B)
            if N_U >= 1:
                if N_U == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(c, d, I_U, Q_U, F1_U, F2_U, f1, f2, F))
                else:
                    rectangles.append(D_U)
    _roots, roots = (sorted(roots, key=lambda r: (r.ax, r.ay)), [])
    for root in _roots:
        roots.extend([root.conjugate(), root])
    if blackbox:
        return roots
    else:
        return [r.as_tuple() for r in roots]