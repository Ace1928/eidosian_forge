from sympy.core.basic import Basic
def assoc(d, k, v):
    d = d.copy()
    d[k] = v
    return d