import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def validate_colorscale(colorscale):
    """Validate the structure, scale values and colors of colorscale."""
    if not isinstance(colorscale, list):
        raise exceptions.PlotlyError('A valid colorscale must be a list.')
    if not all((isinstance(innerlist, list) for innerlist in colorscale)):
        raise exceptions.PlotlyError('A valid colorscale must be a list of lists.')
    colorscale_colors = colorscale_to_colors(colorscale)
    scale_values = colorscale_to_scale(colorscale)
    validate_scale_values(scale_values)
    validate_colors(colorscale_colors)