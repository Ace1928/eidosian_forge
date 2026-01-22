import numpy as np
from ase.io.jsonio import read_json, write_json
def find_height(ldos, current, h, z0=None):
    if z0 is None:
        n = len(ldos) - 2
    else:
        n = int(z0 / h)
    while n >= 0:
        if ldos[n] > current:
            break
        n -= 1
    else:
        return 0.0
    c2, c1 = ldos[n:n + 2]
    return (n + 1 - (current - c1) / (c2 - c1)) * h