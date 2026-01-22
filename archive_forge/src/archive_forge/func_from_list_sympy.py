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
@classmethod
def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
    """
        Convert a list of lists of Expr into a DomainMatrix using construct_domain

        Parameters
        ==========

        nrows: number of rows
        ncols: number of columns
        rows: list of lists

        Returns
        =======

        DomainMatrix containing elements of rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x, y, z
        >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])
        >>> A
        DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])

        See Also
        ========

        sympy.polys.constructor.construct_domain, from_dict_sympy

        """
    assert len(rows) == nrows
    assert all((len(row) == ncols for row in rows))
    items_sympy = [_sympify(item) for row in rows for item in row]
    domain, items_domain = cls.get_domain(items_sympy, **kwargs)
    domain_rows = [[items_domain[ncols * r + c] for c in range(ncols)] for r in range(nrows)]
    return DomainMatrix(domain_rows, (nrows, ncols), domain)