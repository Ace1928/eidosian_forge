from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
Makes a getitem call

        Parameters
        ----------
        obj : ir.Var
            the object being indexed
        index : ir.Var
            the index
        val : ir.Var
            the ty

        Returns
        -------
        res : ir.Expr
            the retrieved value
        