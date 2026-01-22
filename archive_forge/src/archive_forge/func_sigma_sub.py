def sigma_sub(p, q):
    """
    Returns score of a substitution of P with Q.

    (Kondrak 2002: 54)
    """
    return C_sub - delta(p, q) - V(p) - V(q)