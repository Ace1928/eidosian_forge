import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def generate_delaunay(variables, num=10, **kwds):
    """
    Generate a Delaunay triangulation of the D-dimensional
    bounded variable domain given a list of D variables.

    Requires numpy and scipy.

    Args:
        variables: A list of variables, each having a finite
            upper and lower bound.
        num (int): The number of grid points to generate for
            each variable (default=10).
        **kwds: All additional keywords are passed to the
          scipy.spatial.Delaunay constructor.

    Returns:
        A scipy.spatial.Delaunay object.
    """
    linegrids = []
    for v in variables:
        if v.has_lb() and v.has_ub():
            linegrids.append(numpy.linspace(v.lb, v.ub, num))
        else:
            raise ValueError('Variable %s does not have a finite lower and upper bound.')
    points = numpy.vstack(numpy.meshgrid(*linegrids)).reshape(len(variables), -1).T
    return scipy.spatial.Delaunay(points, **kwds)