from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def make_linC_from_linPy(linPy, linPy_to_linC) -> None:
    """Construct a C++ LinOp corresponding to LinPy.

    Children of linPy are retrieved from linPy_to_linC.
    """
    if linPy in linPy_to_linC:
        return
    typ = get_type(linPy)
    shape = cvxcore.IntVector()
    lin_args_vec = cvxcore.ConstLinOpVector()
    for dim in linPy.shape:
        shape.push_back(int(dim))
    for argPy in linPy.args:
        lin_args_vec.push_back(linPy_to_linC[argPy])
    linC = cvxcore.LinOp(typ, shape, lin_args_vec)
    linPy_to_linC[linPy] = linC
    if linPy.data is not None:
        if isinstance(linPy.data, lo.LinOp):
            linC_data = linPy_to_linC[linPy.data]
            linC.set_linOp_data(linC_data)
            linC.set_data_ndim(len(linPy.data.shape))
        else:
            set_linC_data(linC, linPy)