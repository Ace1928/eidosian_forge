from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """
    if is_plotting:
        v = 'Plotting'
    else:
        v = 'Registering plot modes'
    if is_independent:
        n, s = (PlotMode._i_var_max, 'independent')
    else:
        n, s = (PlotMode._d_var_max, 'dependent')
    return '%s with more than %i %s variables is not supported.' % (v, n, s)