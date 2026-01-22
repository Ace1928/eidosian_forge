from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_YAxis(yaxis_title):
    """
    Makes the y-axis for a violin plot.
    """
    yaxis = graph_objs.layout.YAxis(title=yaxis_title, showticklabels=True, autorange=True, ticklen=4, showline=True, zeroline=False, showgrid=False, mirror=False)
    return yaxis