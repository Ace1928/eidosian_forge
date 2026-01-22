import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def create_streamline(x, y, u, v, density=1, angle=math.pi / 9, arrow_scale=0.09, **kwargs):
    """
    Returns data for a streamline plot.

    :param (list|ndarray) x: 1 dimensional, evenly spaced list or array
    :param (list|ndarray) y: 1 dimensional, evenly spaced list or array
    :param (ndarray) u: 2 dimensional array
    :param (ndarray) v: 2 dimensional array
    :param (float|int) density: controls the density of streamlines in
        plot. This is multiplied by 30 to scale similiarly to other
        available streamline functions such as matplotlib.
        Default = 1
    :param (angle in radians) angle: angle of arrowhead. Default = pi/9
    :param (float in [0,1]) arrow_scale: value to scale length of arrowhead
        Default = .09
    :param kwargs: kwargs passed through plotly.graph_objs.Scatter
        for more information on valid kwargs call
        help(plotly.graph_objs.Scatter)

    :rtype (dict): returns a representation of streamline figure.

    Example 1: Plot simple streamline and increase arrow size

    >>> from plotly.figure_factory import create_streamline
    >>> import plotly.graph_objects as go
    >>> import numpy as np
    >>> import math

    >>> # Add data
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.linspace(-3, 3, 100)
    >>> Y, X = np.meshgrid(x, y)
    >>> u = -1 - X**2 + Y
    >>> v = 1 + X - Y**2
    >>> u = u.T  # Transpose
    >>> v = v.T  # Transpose

    >>> # Create streamline
    >>> fig = create_streamline(x, y, u, v, arrow_scale=.1)
    >>> fig.show()

    Example 2: from nbviewer.ipython.org/github/barbagroup/AeroPython

    >>> from plotly.figure_factory import create_streamline
    >>> import numpy as np
    >>> import math

    >>> # Add data
    >>> N = 50
    >>> x_start, x_end = -2.0, 2.0
    >>> y_start, y_end = -1.0, 1.0
    >>> x = np.linspace(x_start, x_end, N)
    >>> y = np.linspace(y_start, y_end, N)
    >>> X, Y = np.meshgrid(x, y)
    >>> ss = 5.0
    >>> x_s, y_s = -1.0, 0.0

    >>> # Compute the velocity field on the mesh grid
    >>> u_s = ss/(2*np.pi) * (X-x_s)/((X-x_s)**2 + (Y-y_s)**2)
    >>> v_s = ss/(2*np.pi) * (Y-y_s)/((X-x_s)**2 + (Y-y_s)**2)

    >>> # Create streamline
    >>> fig = create_streamline(x, y, u_s, v_s, density=2, name='streamline')

    >>> # Add source point
    >>> point = go.Scatter(x=[x_s], y=[y_s], mode='markers',
    ...                    marker_size=14, name='source point')

    >>> fig.add_trace(point) # doctest: +SKIP
    >>> fig.show()
    """
    utils.validate_equal_length(x, y)
    utils.validate_equal_length(u, v)
    validate_streamline(x, y)
    utils.validate_positive_scalars(density=density, arrow_scale=arrow_scale)
    streamline_x, streamline_y = _Streamline(x, y, u, v, density, angle, arrow_scale).sum_streamlines()
    arrow_x, arrow_y = _Streamline(x, y, u, v, density, angle, arrow_scale).get_streamline_arrows()
    streamline = graph_objs.Scatter(x=streamline_x + arrow_x, y=streamline_y + arrow_y, mode='lines', **kwargs)
    data = [streamline]
    layout = graph_objs.Layout(hovermode='closest')
    return graph_objs.Figure(data=data, layout=layout)