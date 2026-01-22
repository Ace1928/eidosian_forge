from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def density_mapbox(data_frame=None, lat=None, lon=None, z=None, hover_name=None, hover_data=None, custom_data=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, opacity=None, zoom=8, center=None, mapbox_style=None, radius=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a Mapbox density map, each row of `data_frame` contributes to the intensity of
    the color of the region around the corresponding point on the map
    """
    return make_figure(args=locals(), constructor=go.Densitymapbox, trace_patch=dict(radius=radius))