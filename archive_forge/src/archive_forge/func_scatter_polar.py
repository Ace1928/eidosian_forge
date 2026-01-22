from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def scatter_polar(data_frame=None, r=None, theta=None, color=None, symbol=None, size=None, hover_name=None, hover_data=None, custom_data=None, text=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, symbol_sequence=None, symbol_map=None, opacity=None, direction='clockwise', start_angle=90, size_max=None, range_r=None, range_theta=None, log_r=False, render_mode='auto', title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a polar scatter plot, each row of `data_frame` is represented by a
    symbol mark in polar coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterpolar)