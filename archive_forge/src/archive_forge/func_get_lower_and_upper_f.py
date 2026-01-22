from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def get_lower_and_upper_f(surface_calculation_options):
    """Get the lower and upper functions defining a surface in the distance-angle space of neighbors.

    Args:
        surface_calculation_options: Options for the surface.

    Returns:
        Dictionary containing the "lower" and "upper" functions for the surface.
    """
    mindist = surface_calculation_options['distance_bounds']['lower']
    maxdist = surface_calculation_options['distance_bounds']['upper']
    minang = surface_calculation_options['angle_bounds']['lower']
    maxang = surface_calculation_options['angle_bounds']['upper']
    if surface_calculation_options['type'] == 'standard_elliptic':
        lower_and_upper_functions = quarter_ellipsis_functions(xx=(mindist, maxang), yy=(maxdist, minang))
    elif surface_calculation_options['type'] == 'standard_diamond':
        deltadist = surface_calculation_options['distance_bounds']['delta']
        deltaang = surface_calculation_options['angle_bounds']['delta']
        lower_and_upper_functions = diamond_functions(xx=(mindist, maxang), yy=(maxdist, minang), x_y0=deltadist, y_x0=deltaang)
    elif surface_calculation_options['type'] == 'standard_spline':
        lower_points = surface_calculation_options['lower_points']
        upper_points = surface_calculation_options['upper_points']
        degree = surface_calculation_options['degree']
        lower_and_upper_functions = spline_functions(lower_points=lower_points, upper_points=upper_points, degree=degree)
    else:
        raise ValueError(f'Surface calculation of type "{surface_calculation_options['type']}" is not implemented')
    return lower_and_upper_functions