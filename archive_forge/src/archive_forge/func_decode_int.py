from typing import Dict, Type, Callable, List
def decode_int(self, x, f):
    f += 1
    newf = x.index(b'e', f)
    n = int(x[f:newf])
    if x[f:f + 2] == b'-0':
        raise ValueError
    elif x[f:f + 1] == b'0' and newf != f + 1:
        raise ValueError
    return (n, newf + 1)