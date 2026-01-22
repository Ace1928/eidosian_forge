import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def validate_streamline(x, y):
    """
    Streamline-specific validations

    Specifically, this checks that x and y are both evenly spaced,
    and that the package numpy is available.

    See FigureFactory.create_streamline() for params

    :raises: (ImportError) If numpy is not available.
    :raises: (PlotlyError) If x is not evenly spaced.
    :raises: (PlotlyError) If y is not evenly spaced.
    """
    if np is False:
        raise ImportError('FigureFactory.create_streamline requires numpy')
    for index in range(len(x) - 1):
        if x[index + 1] - x[index] - (x[1] - x[0]) > 0.0001:
            raise exceptions.PlotlyError('x must be a 1 dimensional, evenly spaced array')
    for index in range(len(y) - 1):
        if y[index + 1] - y[index] - (y[1] - y[0]) > 0.0001:
            raise exceptions.PlotlyError('y must be a 1 dimensional, evenly spaced array')