from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def line_ternary(data_frame=None, a=None, b=None, c=None, color=None, line_dash=None, line_group=None, symbol=None, hover_name=None, hover_data=None, custom_data=None, text=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, line_dash_sequence=None, line_dash_map=None, symbol_sequence=None, symbol_map=None, markers=False, line_shape=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a ternary line plot, each row of `data_frame` is represented as
    vertex of a polyline mark in ternary coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterternary)