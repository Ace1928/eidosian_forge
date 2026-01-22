import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
 Return the number of cones in the tree beyond the known lower bounds.
        if it is zero, then we know the tuple can't be represented in fewer cones.
    