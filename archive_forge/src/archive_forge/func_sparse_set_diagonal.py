from scipy import sparse
import numbers
import numpy as np
def sparse_set_diagonal(X, diag):
    cls = type(X)
    if not isinstance(X, (sparse.lil_matrix, sparse.dia_matrix)):
        X = X.tocoo()
    X.setdiag(diag)
    return cls(X)