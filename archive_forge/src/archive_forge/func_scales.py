import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def scales(key=None, scales={}):
    """Creates and switches between context scales.

    If no key is provided, a new blank context is created.

    If a key is provided for which a context already exists, the existing
    context is set as the current context.

    If a key is provided and no corresponding context exists, a new context is
    created for that key and set as the current context.

    Parameters
    ----------
    key: hashable, optional
        Any variable that can be used as a key for a dictionary
    scales: dictionary
        Dictionary of scales to be used in the new context

    Example
    -------

        >>> scales(scales={
        >>>    'x': Keep,
        >>>    'color': ColorScale(min=0, max=1)
        >>> })

    This creates a new scales context, where the 'x' scale is kept from the
    previous context, the 'color' scale is an instance of ColorScale
    provided by the user. Other scales, potentially needed such as the 'y'
    scale in the case of a line chart will be created on the fly when
    needed.

    Notes
    -----
    Every call to the function figure triggers a call to scales.

    The `scales` parameter is ignored if the `key` argument is not Keep and
    context scales already exist for that key.
    """
    old_ctxt = _context['scales']
    if key is None:
        _context['scales'] = {_get_attribute_dimension(k): scales[k] if scales[k] is not Keep else old_ctxt[_get_attribute_dimension(k)] for k in scales}
    else:
        if key not in _context['scale_registry']:
            _context['scale_registry'][key] = {_get_attribute_dimension(k): scales[k] if scales[k] is not Keep else old_ctxt[_get_attribute_dimension(k)] for k in scales}
        _context['scales'] = _context['scale_registry'][key]