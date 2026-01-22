def sigma_exp(p, q):
    """
    Returns score of an expansion/compression.

    (Kondrak 2002: 54)
    """
    q1 = q[0]
    q2 = q[1]
    return C_exp - delta(p, q1) - delta(p, q2) - V(p) - max(V(q1), V(q2))