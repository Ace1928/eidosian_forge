from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_violin_rugplot(vals, pdf_max, distance, color='#1f77b4'):
    """
    Returns a rugplot fig for a violin plot.
    """
    return graph_objs.Scatter(y=vals, x=[-pdf_max - distance] * len(vals), marker=graph_objs.scatter.Marker(color=color, symbol='line-ew-open'), mode='markers', name='', showlegend=False, hoverinfo='y')