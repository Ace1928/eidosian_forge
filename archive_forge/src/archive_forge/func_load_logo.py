import os
from unittest import SkipTest
import param
from IPython.core.completer import IPCompleter
from IPython.display import HTML, publish_display_data
from param import ipython as param_ext
import holoviews as hv
from ..core.dimension import LabelledData
from ..core.options import Store
from ..core.tree import AttrTree
from ..element.comparison import ComparisonTestCase
from ..plotting.renderer import Renderer
from ..util import extension
from .display_hooks import display, png_display, pprint_display, svg_display
from .magics import load_magics
@classmethod
def load_logo(cls, logo=False, bokeh_logo=False, mpl_logo=False, plotly_logo=False):
    """
        Allow to display Holoviews' logo and the plotting extensions' logo.
        """
    import jinja2
    templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    jinjaEnv = jinja2.Environment(loader=templateLoader)
    template = jinjaEnv.get_template('load_notebook.html')
    html = template.render({'logo': logo, 'bokeh_logo': bokeh_logo, 'mpl_logo': mpl_logo, 'plotly_logo': plotly_logo})
    publish_display_data(data={'text/html': html})