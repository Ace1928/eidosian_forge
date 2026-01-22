from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def scatter_ternary(data_frame=None, a=None, b=None, c=None, color=None, symbol=None, size=None, text=None, hover_name=None, hover_data=None, custom_data=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, symbol_sequence=None, symbol_map=None, opacity=None, size_max=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a ternary scatter plot, each row of `data_frame` is represented by a
    symbol mark in ternary coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterternary)