from snappy.pari import pari
def solve_right(self, b):
    """
        Return a vector v for which A v = b.

        >>> A = Matrix(2, 2, range(4))
        >>> A.solve_right([6, 8])
        [-5, 6]
        """
    return self.solve(b)