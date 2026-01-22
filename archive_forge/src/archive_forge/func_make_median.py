from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_median(q2):
    """
    Formats the 'median' hovertext for a violin plot.
    """
    return graph_objs.Scatter(x=[0], y=[q2], text=['median: ' + '{:0.2f}'.format(q2)], mode='markers', marker=dict(symbol='square', color='rgb(255,255,255)'), hoverinfo='text')