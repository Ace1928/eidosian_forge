from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def line_mapbox(data_frame=None, lat=None, lon=None, color=None, text=None, hover_name=None, hover_data=None, custom_data=None, line_group=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, zoom=8, center=None, mapbox_style=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a Mapbox line plot, each row of `data_frame` is represented as
    vertex of a polyline mark on a Mapbox map.
    """
    return make_figure(args=locals(), constructor=go.Scattermapbox)