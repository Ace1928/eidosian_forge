from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def build_lin_op_tree(root_linPy, linPy_to_linC) -> None:
    """Construct C++ LinOp tree from Python LinOp tree.

    Constructed C++ linOps are stored in the linPy_to_linC dict,
    which maps Python linOps to their corresponding C++ linOps.

    Parameters
    ----------
        linPy_to_linC: a dict for memoizing construction and storing
            the C++ LinOps
    """
    bfs_stack = [root_linPy]
    post_order_stack = []
    while bfs_stack:
        linPy = bfs_stack.pop()
        if linPy not in linPy_to_linC:
            post_order_stack.append(linPy)
            for arg in linPy.args:
                bfs_stack.append(arg)
            if isinstance(linPy.data, lo.LinOp):
                bfs_stack.append(linPy.data)
    while post_order_stack:
        linPy = post_order_stack.pop()
        make_linC_from_linPy(linPy, linPy_to_linC)