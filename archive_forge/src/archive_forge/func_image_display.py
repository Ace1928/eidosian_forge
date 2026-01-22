import sys
import traceback
from contextlib import contextmanager
from functools import wraps
import IPython
from IPython import get_ipython
from IPython.display import HTML
import holoviews as hv
from ..core import (
from ..core.io import FileArchive
from ..core.options import AbbreviatedException, SkipRendering, Store, StoreOptions
from ..core.traversal import unique_dimkeys
from ..core.util import mimebundle_to_html
from ..plotting import Plot
from ..plotting.renderer import MIME_TYPES
from ..util.settings import OutputSettings
from .magics import OptsMagic, OutputMagic
def image_display(element, max_frames, fmt):
    """
    Used to render elements to an image format (svg or png) if requested
    in the display formats.
    """
    if fmt not in Store.display_formats:
        return None
    info = process_object(element)
    if info:
        display(HTML(info))
        return
    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None
    renderer = Store.renderers[backend]
    plot = renderer.get_plot(element)
    if fmt not in renderer.param.objects('existing')['fig'].objects:
        return None
    data, info = renderer(plot, fmt=fmt)
    return ({info['mime_type']: data}, {})