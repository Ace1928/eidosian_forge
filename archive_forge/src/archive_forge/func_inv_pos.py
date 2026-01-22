from cvxpy.atoms.elementwise.power import power
def inv_pos(x):
    """:math:`x^{-1}` for :math:`x > 0`.
    """
    return power(x, -1)