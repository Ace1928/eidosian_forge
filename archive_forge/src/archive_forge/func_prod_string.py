import numpy as np
def prod_string(v, name):
    v = np.abs(v)
    if v != 1:
        ss = str(v) + ' * ' + name
    else:
        ss = name
    return ss