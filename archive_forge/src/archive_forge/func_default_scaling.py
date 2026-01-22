from scipy.sparse import eye as speye
from .projections import projections
from .qp_subproblem import modified_dogleg, projected_cg, box_intersections
import numpy as np
from numpy.linalg import norm
def default_scaling(x):
    n, = np.shape(x)
    return speye(n)