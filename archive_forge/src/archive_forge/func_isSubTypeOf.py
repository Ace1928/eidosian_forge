import sys
from pyasn1.type import error
def isSubTypeOf(self, otherConstraint):
    return otherConstraint is self or not self or otherConstraint == self or (otherConstraint in self._valueMap)