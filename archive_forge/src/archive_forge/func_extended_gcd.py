from rsa._compat import zip
def extended_gcd(a, b):
    """Returns a tuple (r, i, j) such that r = gcd(a, b) = ia + jb
    """
    x = 0
    y = 1
    lx = 1
    ly = 0
    oa = a
    ob = b
    while b != 0:
        q = a // b
        a, b = (b, a % b)
        x, lx = (lx - q * x, x)
        y, ly = (ly - q * y, y)
    if lx < 0:
        lx += ob
    if ly < 0:
        ly += oa
    return (a, lx, ly)