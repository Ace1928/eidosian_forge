import collections
import numbers
def solution_value(self):
    """Value of this linear expr, using the solution_value of its vars."""
    coeffs = self.GetCoeffs()
    return sum((var.solution_value() * coeff for var, coeff in coeffs.items()))