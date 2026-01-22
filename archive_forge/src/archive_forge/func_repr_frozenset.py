import builtins
from itertools import islice
from _thread import get_ident
def repr_frozenset(self, x, level):
    if not x:
        return 'frozenset()'
    x = _possibly_sorted(x)
    return self._repr_iterable(x, level, 'frozenset({', '})', self.maxfrozenset)