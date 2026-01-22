import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator
def create_annotated_heatmap(z, x=None, y=None, annotation_text=None, colorscale='Plasma', font_colors=None, showscale=False, reversescale=False, **kwargs):
    """
    **deprecated**, use instead
    :func:`plotly.express.imshow`.

    Function that creates annotated heatmaps

    This function adds annotations to each cell of the heatmap.

    :param (list[list]|ndarray) z: z matrix to create heatmap.
    :param (list) x: x axis labels.
    :param (list) y: y axis labels.
    :param (list[list]|ndarray) annotation_text: Text strings for
        annotations. Should have the same dimensions as the z matrix. If no
        text is added, the values of the z matrix are annotated. Default =
        z matrix values.
    :param (list|str) colorscale: heatmap colorscale.
    :param (list) font_colors: List of two color strings: [min_text_color,
        max_text_color] where min_text_color is applied to annotations for
        heatmap values < (max_value - min_value)/2. If font_colors is not
        defined, the colors are defined logically as black or white
        depending on the heatmap's colorscale.
    :param (bool) showscale: Display colorscale. Default = False
    :param (bool) reversescale: Reverse colorscale. Default = False
    :param kwargs: kwargs passed through plotly.graph_objs.Heatmap.
        These kwargs describe other attributes about the annotated Heatmap
        trace such as the colorscale. For more information on valid kwargs
        call help(plotly.graph_objs.Heatmap)

    Example 1: Simple annotated heatmap with default configuration

    >>> import plotly.figure_factory as ff

    >>> z = [[0.300000, 0.00000, 0.65, 0.300000],
    ...      [1, 0.100005, 0.45, 0.4300],
    ...      [0.300000, 0.00000, 0.65, 0.300000],
    ...      [1, 0.100005, 0.45, 0.00000]]

    >>> fig = ff.create_annotated_heatmap(z)
    >>> fig.show()
    """
    font_colors = font_colors if font_colors is not None else []
    validate_annotated_heatmap(z, x, y, annotation_text)
    colorscale_validator = ColorscaleValidator()
    colorscale = colorscale_validator.validate_coerce(colorscale)
    annotations = _AnnotatedHeatmap(z, x, y, annotation_text, colorscale, font_colors, reversescale, **kwargs).make_annotations()
    if x or y:
        trace = dict(type='heatmap', z=z, x=x, y=y, colorscale=colorscale, showscale=showscale, reversescale=reversescale, **kwargs)
        layout = dict(annotations=annotations, xaxis=dict(ticks='', dtick=1, side='top', gridcolor='rgb(0, 0, 0)'), yaxis=dict(ticks='', dtick=1, ticksuffix='  '))
    else:
        trace = dict(type='heatmap', z=z, colorscale=colorscale, showscale=showscale, reversescale=reversescale, **kwargs)
        layout = dict(annotations=annotations, xaxis=dict(ticks='', side='top', gridcolor='rgb(0, 0, 0)', showticklabels=False), yaxis=dict(ticks='', ticksuffix='  ', showticklabels=False))
    data = [trace]
    return graph_objs.Figure(data=data, layout=layout)