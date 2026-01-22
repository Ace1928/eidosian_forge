from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def bar_polar(data_frame=None, r=None, theta=None, color=None, pattern_shape=None, hover_name=None, hover_data=None, custom_data=None, base=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, pattern_shape_sequence=None, pattern_shape_map=None, range_color=None, color_continuous_midpoint=None, barnorm=None, barmode='relative', direction='clockwise', start_angle=90, range_r=None, range_theta=None, log_r=False, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a polar bar plot, each row of `data_frame` is represented as a wedge
    mark in polar coordinates.
    """
    return make_figure(args=locals(), constructor=go.Barpolar, layout_patch=dict(barnorm=barnorm, barmode=barmode))