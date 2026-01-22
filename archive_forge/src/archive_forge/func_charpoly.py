from functools import reduce
from typing import Union as tUnion, Tuple as tTuple
from sympy.core.sympify import _sympify
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import (DMNonSquareMatrixError, DMShapeError,
from .ddm import DDM
from .sdm import SDM
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
def charpoly(self):
    """
        Returns the coefficients of the characteristic polynomial
        of the DomainMatrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the DomainMatrix.

        Returns
        =======

        list
            coefficients of the characteristic polynomial

        Raises
        ======

        DMNonSquareMatrixError
            If the DomainMatrix is not a not Square DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.charpoly()
        [1, -5, -2]

        """
    m, n = self.shape
    if m != n:
        raise DMNonSquareMatrixError('not square')
    return self.rep.charpoly()