from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def choropleth_mapbox(data_frame=None, geojson=None, featureidkey=None, locations=None, color=None, hover_name=None, hover_data=None, custom_data=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, opacity=None, zoom=8, center=None, mapbox_style=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a Mapbox choropleth map, each row of `data_frame` is represented by a
    colored region on a Mapbox map.
    """
    return make_figure(args=locals(), constructor=go.Choroplethmapbox)