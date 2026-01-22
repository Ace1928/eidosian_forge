from typing import List, Tuple
import numpy as np
import scipy as scipy
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
Quadratic of piecewise affine if x is PWL and y is constant.
        