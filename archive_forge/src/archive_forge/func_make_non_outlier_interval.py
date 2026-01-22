from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_non_outlier_interval(d1, d2):
    """
    Returns the scatterplot fig of most of a violin plot.
    """
    return graph_objs.Scatter(x=[0, 0], y=[d1, d2], name='', mode='lines', line=graph_objs.scatter.Line(width=1.5, color='rgb(0,0,0)'))