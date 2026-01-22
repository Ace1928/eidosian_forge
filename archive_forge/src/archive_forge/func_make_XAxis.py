from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_XAxis(xaxis_title, xaxis_range):
    """
    Makes the x-axis for a violin plot.
    """
    xaxis = graph_objs.layout.XAxis(title=xaxis_title, range=xaxis_range, showgrid=False, zeroline=False, showline=False, mirror=False, ticks='', showticklabels=False)
    return xaxis