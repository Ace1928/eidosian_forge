from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util
def piecewise_nd(tri, values, input=None, output=None, bound='eq', repn='cc'):
    """
    Models a multi-variate piecewise linear function.

    This function takes a D-dimensional triangulation and a
    list of function values associated with the points of
    the triangulation and transforms this input data into a
    block of variables and constraints that enforce a
    piecewise linear relationship between an D-dimensional
    vector of input variable and a single output
    variable. In the general case, this transformation
    requires the use of discrete decision variables.

    Args:
        tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:

              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
        values (numpy.array): An (npoints,) shaped array of
            the values of the piecewise function at each of
            coordinates in the triangulation points array.
        input: A D-length list of variables or expressions
            bound as the inputs of the piecewise function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
        repn (str): The type of piecewise representation to
            use. Can be one of:

                - 'cc': convex combination

    Returns:
        TransformedPiecewiseLinearFunctionND: a block
            containing any new variables, constraints, and
            other components used by the piecewise
            representation
    """
    transform = None
    try:
        transform = registered_transforms[repn]
    except KeyError:
        raise ValueError("Keyword assignment repn='%s' is not valid. Must be one of: %s" % (repn, str(sorted(registered_transforms.keys()))))
    assert transform is not None
    func = PiecewiseLinearFunctionND(tri, values)
    return transform(func, input=input, output=output, bound=bound)