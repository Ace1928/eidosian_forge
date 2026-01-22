import builtins
from itertools import islice
from _thread import get_ident
def repr_int(self, x, level):
    s = builtins.repr(x)
    if len(s) > self.maxlong:
        i = max(0, (self.maxlong - 3) // 2)
        j = max(0, self.maxlong - 3 - i)
        s = s[:i] + self.fillvalue + s[len(s) - j:]
    return s