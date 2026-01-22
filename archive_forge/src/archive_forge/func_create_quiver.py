import math
from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def create_quiver(x, y, u, v, scale=0.1, arrow_scale=0.3, angle=math.pi / 9, scaleratio=None, **kwargs):
    """
    Returns data for a quiver plot.

    :param (list|ndarray) x: x coordinates of the arrow locations
    :param (list|ndarray) y: y coordinates of the arrow locations
    :param (list|ndarray) u: x components of the arrow vectors
    :param (list|ndarray) v: y components of the arrow vectors
    :param (float in [0,1]) scale: scales size of the arrows(ideally to
        avoid overlap). Default = .1
    :param (float in [0,1]) arrow_scale: value multiplied to length of barb
        to get length of arrowhead. Default = .3
    :param (angle in radians) angle: angle of arrowhead. Default = pi/9
    :param (positive float) scaleratio: the ratio between the scale of the y-axis
        and the scale of the x-axis (scale_y / scale_x). Default = None, the
        scale ratio is not fixed.
    :param kwargs: kwargs passed through plotly.graph_objs.Scatter
        for more information on valid kwargs call
        help(plotly.graph_objs.Scatter)

    :rtype (dict): returns a representation of quiver figure.

    Example 1: Trivial Quiver

    >>> from plotly.figure_factory import create_quiver
    >>> import math

    >>> # 1 Arrow from (0,0) to (1,1)
    >>> fig = create_quiver(x=[0], y=[0], u=[1], v=[1], scale=1)
    >>> fig.show()


    Example 2: Quiver plot using meshgrid

    >>> from plotly.figure_factory import create_quiver

    >>> import numpy as np
    >>> import math

    >>> # Add data
    >>> x,y = np.meshgrid(np.arange(0, 2, .2), np.arange(0, 2, .2))
    >>> u = np.cos(x)*y
    >>> v = np.sin(x)*y

    >>> #Create quiver
    >>> fig = create_quiver(x, y, u, v)
    >>> fig.show()


    Example 3: Styling the quiver plot

    >>> from plotly.figure_factory import create_quiver
    >>> import numpy as np
    >>> import math

    >>> # Add data
    >>> x, y = np.meshgrid(np.arange(-np.pi, math.pi, .5),
    ...                    np.arange(-math.pi, math.pi, .5))
    >>> u = np.cos(x)*y
    >>> v = np.sin(x)*y

    >>> # Create quiver
    >>> fig = create_quiver(x, y, u, v, scale=.2, arrow_scale=.3, angle=math.pi/6,
    ...                     name='Wind Velocity', line=dict(width=1))

    >>> # Add title to layout
    >>> fig.update_layout(title='Quiver Plot') # doctest: +SKIP
    >>> fig.show()


    Example 4: Forcing a fix scale ratio to maintain the arrow length

    >>> from plotly.figure_factory import create_quiver
    >>> import numpy as np

    >>> # Add data
    >>> x,y = np.meshgrid(np.arange(0.5, 3.5, .5), np.arange(0.5, 4.5, .5))
    >>> u = x
    >>> v = y
    >>> angle = np.arctan(v / u)
    >>> norm = 0.25
    >>> u = norm * np.cos(angle)
    >>> v = norm * np.sin(angle)

    >>> # Create quiver with a fix scale ratio
    >>> fig = create_quiver(x, y, u, v, scale = 1, scaleratio = 0.5)
    >>> fig.show()
    """
    utils.validate_equal_length(x, y, u, v)
    utils.validate_positive_scalars(arrow_scale=arrow_scale, scale=scale)
    if scaleratio is None:
        quiver_obj = _Quiver(x, y, u, v, scale, arrow_scale, angle)
    else:
        quiver_obj = _Quiver(x, y, u, v, scale, arrow_scale, angle, scaleratio)
    barb_x, barb_y = quiver_obj.get_barbs()
    arrow_x, arrow_y = quiver_obj.get_quiver_arrows()
    quiver_plot = graph_objs.Scatter(x=barb_x + arrow_x, y=barb_y + arrow_y, mode='lines', **kwargs)
    data = [quiver_plot]
    if scaleratio is None:
        layout = graph_objs.Layout(hovermode='closest')
    else:
        layout = graph_objs.Layout(hovermode='closest', yaxis=dict(scaleratio=scaleratio, scaleanchor='x'))
    return graph_objs.Figure(data=data, layout=layout)