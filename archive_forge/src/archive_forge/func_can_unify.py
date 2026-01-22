from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
def can_unify(self, other):
    if other.is_var():
        return [(other, self)]
    if other.is_function():
        sa = self._res.can_unify(other.res())
        sd = self._dir.can_unify(other.dir())
        if sa is not None and sd is not None:
            sb = self._arg.substitute(sa).can_unify(other.arg().substitute(sa))
            if sb is not None:
                return sa + sb
    return None