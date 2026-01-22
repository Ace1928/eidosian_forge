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
def load_nb(cls, inline=True):
    """
        Loads the plotly notebook resources.
        """
    import panel.models.plotly
    cls._loaded = True
    if 'plotly' not in getattr(pn.extension, '_loaded_extensions', ['plotly']):
        pn.extension._loaded_extensions.append('plotly')