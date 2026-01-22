from typing import Dict, Type, Callable, List
def decode_string(self, x, f):
    colon = x.index(b':', f)
    n = int(x[f:colon])
    if x[f:f + 1] == b'0' and colon != f + 1:
        raise ValueError
    colon += 1
    return (x[colon:colon + n], colon + n)