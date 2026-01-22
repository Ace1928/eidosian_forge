import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def prettydict(d):
    """ Print keys of a dictionary with children (expected to be a Sequence) indented underneath.

    Used for printing out trees of second order cones to represent weighted geometric means.

    """
    keys = sorted(list(d.keys()), key=get_max_denom, reverse=True)
    result = ''
    for tup in keys:
        children = sorted(d[tup], key=get_max_denom, reverse=False)
        result += prettytuple(tup) + '\n'
        for child in children:
            result += '  ' + prettytuple(child) + '\n'
    return result