import base64
from io import BytesIO
import panel as pn
import param
from param.parameterized import bothmethod
from ...core import HoloMap
from ...core.options import Store
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .callbacks import callbacks
from .util import clean_internal_figure_properties
@classmethod
def plot_options(cls, obj, percent_size):
    factor = percent_size / 100.0
    obj = obj.last if isinstance(obj, HoloMap) else obj
    plot = Store.registry[cls.backend].get(type(obj), None)
    options = plot.lookup_options(obj, 'plot').options
    width = options.get('width', plot.width) * factor
    height = options.get('height', plot.height) * factor
    return dict(options, width=int(width), height=int(height))