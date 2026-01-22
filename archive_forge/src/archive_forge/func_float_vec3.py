from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
def float_vec3(f):

    def inner(*args):
        v = f(*args)
        return (float(v[0]), float(v[1]), float(v[2]))
    return inner