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
def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
    """

        Parameters
        ==========

        nrows: number of rows
        ncols: number of cols
        elemsdict: dict of dicts containing non-zero elements of the DomainMatrix

        Returns
        =======

        DomainMatrix containing elements of elemsdict

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x,y,z
        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}
        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)
        >>> A
        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])

        See Also
        ========

        from_list_sympy

        """
    if not all((0 <= r < nrows for r in elemsdict)):
        raise DMBadInputError('Row out of range')
    if not all((0 <= c < ncols for row in elemsdict.values() for c in row)):
        raise DMBadInputError('Column out of range')
    items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
    domain, items_domain = cls.get_domain(items_sympy, **kwargs)
    idx = 0
    items_dict = {}
    for i, row in elemsdict.items():
        items_dict[i] = {}
        for j in row:
            items_dict[i][j] = items_domain[idx]
            idx += 1
    return DomainMatrix(items_dict, (nrows, ncols), domain)