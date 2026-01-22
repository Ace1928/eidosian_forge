from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
def simple_root(self, i):
    """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        This method returns the ith simple root for E_n.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("E6")
        >>> c.simple_root(2)
        [1, 1, 0, 0, 0, 0, 0, 0]
        """
    n = self.n
    if i == 1:
        root = [-0.5] * 8
        root[0] = 0.5
        root[7] = 0.5
        return root
    elif i == 2:
        root = [0] * 8
        root[1] = 1
        root[0] = 1
        return root
    else:
        if i in (7, 8) and n == 6:
            raise ValueError('E6 only has six simple roots!')
        if i == 8 and n == 7:
            raise ValueError('E7 has only 7 simple roots!')
        return self.basic_root(i - 3, i - 2)