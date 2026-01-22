from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def scatter_geo(data_frame=None, lat=None, lon=None, locations=None, locationmode=None, geojson=None, featureidkey=None, color=None, text=None, symbol=None, facet_row=None, facet_col=None, facet_col_wrap=0, facet_row_spacing=None, facet_col_spacing=None, hover_name=None, hover_data=None, custom_data=None, size=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, symbol_sequence=None, symbol_map=None, opacity=None, size_max=None, projection=None, scope=None, center=None, fitbounds=None, basemap_visible=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a geographic scatter plot, each row of `data_frame` is represented
    by a symbol mark on a map.
    """
    return make_figure(args=locals(), constructor=go.Scattergeo, trace_patch=dict(locationmode=locationmode))