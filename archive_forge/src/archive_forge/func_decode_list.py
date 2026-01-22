from typing import Dict, Type, Callable, List
def decode_list(self, x, f):
    r, f = ([], f + 1)
    while x[f:f + 1] != b'e':
        v, f = self.decode_func[x[f:f + 1]](x, f)
        r.append(v)
    if self.yield_tuples:
        r = tuple(r)
    return (r, f + 1)