from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def print_iteration_header():
    print('{:^15}{:^15}{:^15}{:^15}{:^15}'.format('Iteration', 'Max residual', 'Max BC residual', 'Total nodes', 'Nodes added'))