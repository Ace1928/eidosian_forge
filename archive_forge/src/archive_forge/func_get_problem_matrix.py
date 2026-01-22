from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def get_problem_matrix(linOps, var_length, id_to_col, param_to_size, param_to_col, constr_length, canon_backend: str | None=None):
    """
    Builds a sparse representation of the problem data.

    Parameters
    ----------
        linOps: A list of python linOp trees representing an affine expression
        var_length: The total length of the variables.
        id_to_col: A map from variable id to column offset.
        param_to_size: A map from parameter id to parameter size.
        param_to_col: A map from parameter id to column in tensor.
        constr_length: Summed sizes of constraints input.
        canon_backend :
            'CPP' (default) | 'SCIPY'
            Specifies which backend to use for canonicalization, which can affect
            compilation time. Defaults to None, i.e., selecting the default backend.

    Returns
    -------
        A sparse (CSC) matrix with constr_length * (var_length + 1) rows and
        param_size + 1 columns (where param_size is the length of the
        parameter vector).
    """
    default_canon_backend = get_default_canon_backend()
    canon_backend = default_canon_backend if not canon_backend else canon_backend
    if canon_backend == s.CPP_CANON_BACKEND:
        lin_vec = cvxcore.ConstLinOpVector()
        id_to_col_C = cvxcore.IntIntMap()
        for id, col in id_to_col.items():
            id_to_col_C[int(id)] = int(col)
        param_to_size_C = cvxcore.IntIntMap()
        for id, size in param_to_size.items():
            param_to_size_C[int(id)] = int(size)
        linPy_to_linC = {}
        for lin in linOps:
            build_lin_op_tree(lin, linPy_to_linC)
            tree = linPy_to_linC[lin]
            lin_vec.push_back(tree)
        problemData = cvxcore.build_matrix(lin_vec, int(var_length), id_to_col_C, param_to_size_C, s.get_num_threads())
        tensor_V = {}
        tensor_I = {}
        tensor_J = {}
        for param_id, size in param_to_size.items():
            tensor_V[param_id] = []
            tensor_I[param_id] = []
            tensor_J[param_id] = []
            problemData.param_id = param_id
            for i in range(size):
                problemData.vec_idx = i
                prob_len = problemData.getLen()
                tensor_V[param_id].append(problemData.getV(prob_len))
                tensor_I[param_id].append(problemData.getI(prob_len))
                tensor_J[param_id].append(problemData.getJ(prob_len))
        V = []
        I = []
        J = []
        param_size_plus_one = 0
        for param_id, col in param_to_col.items():
            size = param_to_size[param_id]
            param_size_plus_one += size
            for i in range(size):
                V.append(tensor_V[param_id][i])
                I.append(tensor_I[param_id][i] + tensor_J[param_id][i] * constr_length)
                J.append(tensor_J[param_id][i] * 0 + (i + col))
        V = np.concatenate(V)
        I = np.concatenate(I)
        J = np.concatenate(J)
        output_shape = (np.int64(constr_length) * np.int64(var_length + 1), param_size_plus_one)
        A = sp.csc_matrix((V, (I, J)), shape=output_shape)
        return A
    elif canon_backend in {s.SCIPY_CANON_BACKEND, s.RUST_CANON_BACKEND, s.NUMPY_CANON_BACKEND}:
        param_size_plus_one = sum(param_to_size.values())
        output_shape = (np.int64(constr_length) * np.int64(var_length + 1), param_size_plus_one)
        if len(linOps) > 0:
            backend = CanonBackend.get_backend(canon_backend, id_to_col, param_to_size, param_to_col, param_size_plus_one, var_length)
            A_py = backend.build_matrix(linOps)
        else:
            A_py = sp.csc_matrix(((), ((), ())), output_shape)
        assert A_py.shape == output_shape
        return A_py
    else:
        raise ValueError(f'Unknown backend: {canon_backend}')