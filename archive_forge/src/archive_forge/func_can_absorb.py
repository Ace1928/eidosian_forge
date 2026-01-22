from __future__ import print_function
from patsy.util import no_pickling
def can_absorb(self, other):
    return len(self.efactors) - len(other.efactors) == 1 and self.efactors.issuperset(other.efactors)