import sys
from copy import deepcopy
from functools import partial
from operator import mul, truediv
def dominates(self, other):
    self_violates_constraints = _violates_constraint(self)
    other_violates_constraints = _violates_constraint(other)
    if self_violates_constraints and other_violates_constraints:
        return False
    elif self_violates_constraints:
        return False
    elif other_violates_constraints:
        return True
    return super(ConstrainedFitness, self).dominates(other)