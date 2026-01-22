from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def get_default_canon_backend() -> str:
    """
    Returns the default canonicalization backend, which can be set globally using an
    environment variable.
    """
    return os.environ.get('CVXPY_DEFAULT_CANON_BACKEND', s.DEFAULT_CANON_BACKEND)