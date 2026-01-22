from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def print_iteration_progress(iteration, residual, bc_residual, total_nodes, nodes_added):
    print('{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}'.format(iteration, residual, bc_residual, total_nodes, nodes_added))