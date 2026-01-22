import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def init_notebook_mode(connected=False):
    """
    Initialize plotly.js in the browser if it hasn't been loaded into the DOM
    yet. This is an idempotent method and can and should be called from any
    offline methods that require plotly.js to be loaded into the notebook dom.

    Keyword arguments:

    connected (default=False) -- If True, the plotly.js library will be loaded
    from an online CDN. If False, the plotly.js library will be loaded locally
    from the plotly python package

    Use `connected=True` if you want your notebooks to have smaller file sizes.
    In the case where `connected=False`, the entirety of the plotly.js library
    will be loaded into the notebook, which will result in a file-size increase
    of a couple megabytes. Additionally, because the library will be downloaded
    from the web, you and your viewers must be connected to the internet to be
    able to view charts within this notebook.

    Use `connected=False` if you want you and your collaborators to be able to
    create and view these charts regardless of the availability of an internet
    connection. This is the default option since it is the most predictable.
    Note that under this setting the library will be included inline inside
    your notebook, resulting in much larger notebook sizes compared to the case
    where `connected=True`.
    """
    import plotly.io as pio
    ipython = get_module('IPython')
    if not ipython:
        raise ImportError('`iplot` can only run inside an IPython Notebook.')
    if connected:
        pio.renderers.default = 'plotly_mimetype+notebook_connected'
    else:
        pio.renderers.default = 'plotly_mimetype+notebook'
    pio.renderers._activate_pending_renderers()