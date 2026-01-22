from .functions import defun, defun_wrapped
def search_supergood_block(ctx, n, fp_tolerance):
    """To use for n>400 000 000"""
    sb = sure_number_block(ctx, n)
    number_goodblocks = 0
    m2 = n - 1
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf = [t]
    Vf = [v]
    while b < 0:
        m2 += 1
        t, v, b = compute_triple_tvb(ctx, m2)
        Tf.append(t)
        Vf.append(v)
    goodpoints = [m2]
    T = [t]
    V = [v]
    while number_goodblocks < 2 * sb:
        m2 += 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.append(t)
        V.append(v)
        while b < 0:
            m2 += 1
            t, v, b = compute_triple_tvb(ctx, m2)
            T.append(t)
            V.append(v)
        goodpoints.append(m2)
        zn = len(T) - 1
        A, B, separated = separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
        Tf.pop()
        Tf.extend(A)
        Vf.pop()
        Vf.extend(B)
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    number_goodblocks = 0
    m2 = n - 2
    t, v, b = compute_triple_tvb(ctx, m2)
    Tf.insert(0, t)
    Vf.insert(0, v)
    while b < 0:
        m2 -= 1
        t, v, b = compute_triple_tvb(ctx, m2)
        Tf.insert(0, t)
        Vf.insert(0, v)
    goodpoints.insert(0, m2)
    T = [t]
    V = [v]
    while number_goodblocks < 2 * sb:
        m2 -= 1
        t, v, b = compute_triple_tvb(ctx, m2)
        T.insert(0, t)
        V.insert(0, v)
        while b < 0:
            m2 -= 1
            t, v, b = compute_triple_tvb(ctx, m2)
            T.insert(0, t)
            V.insert(0, v)
        goodpoints.insert(0, m2)
        zn = len(T) - 1
        A, B, separated = separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
        A.pop()
        Tf = A + Tf
        B.pop()
        Vf = B + Vf
        if separated:
            number_goodblocks += 1
        else:
            number_goodblocks = 0
        T = [t]
        V = [v]
    r = goodpoints[2 * sb]
    lg = len(goodpoints)
    s = goodpoints[lg - 2 * sb - 1]
    tr, vr, br = compute_triple_tvb(ctx, r)
    ar = Tf.index(tr)
    ts, vs, bs = compute_triple_tvb(ctx, s)
    as1 = Tf.index(ts)
    T = Tf[ar:as1 + 1]
    V = Vf[ar:as1 + 1]
    zn = s - r
    A, B, separated = separate_zeros_in_block(ctx, zn, T, V, limitloop=ITERATION_LIMIT, fp_tolerance=fp_tolerance)
    if separated:
        return (n - r - 1, [r, s], A, B)
    q = goodpoints[sb]
    lg = len(goodpoints)
    t = goodpoints[lg - sb - 1]
    tq, vq, bq = compute_triple_tvb(ctx, q)
    aq = Tf.index(tq)
    tt, vt, bt = compute_triple_tvb(ctx, t)
    at = Tf.index(tt)
    T = Tf[aq:at + 1]
    V = Vf[aq:at + 1]
    return (n - q - 1, [q, t], T, V)