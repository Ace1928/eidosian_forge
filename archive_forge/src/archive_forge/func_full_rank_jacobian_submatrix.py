from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def full_rank_jacobian_submatrix(self):
    return self.jacobian().matrix_from_rows_and_columns(self.exact_edges, self.var_edges)