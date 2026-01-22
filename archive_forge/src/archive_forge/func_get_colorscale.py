import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def get_colorscale(name):
    """
    Returns the colorscale for a given name. See `named_colorscales` for the
    built-in colorscales.
    """
    from _plotly_utils.basevalidators import ColorscaleValidator
    if not isinstance(name, str):
        raise exceptions.PlotlyError('Name argument have to be a string.')
    name = name.lower()
    if name[-2:] == '_r':
        should_reverse = True
        name = name[:-2]
    else:
        should_reverse = False
    if name in ColorscaleValidator('', '').named_colorscales:
        colorscale = ColorscaleValidator('', '').named_colorscales[name]
    else:
        raise exceptions.PlotlyError(f'Colorscale {name} is not a built-in scale.')
    if should_reverse:
        colorscale = colorscale[::-1]
    return make_colorscale(colorscale)