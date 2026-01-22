import inspect
import textwrap
import param
import panel as _pn
import holoviews as _hv
from holoviews import Store, render  # noqa
from .converter import HoloViewsConverter
from .interactive import Interactive
from .ui import explorer  # noqa
from .utilities import hvplot_extension, output, save, show # noqa
from .plotting import (hvPlot, hvPlotTabular,  # noqa
def post_patch(extension='bokeh', logo=False):
    if extension and (not getattr(_hv.extension, '_loaded', False)):
        hvplot_extension(extension, logo=logo)