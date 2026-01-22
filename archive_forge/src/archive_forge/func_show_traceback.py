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
def show_traceback():
    """
    Display the full traceback after an abbreviated traceback has occurred.
    """
    from .display_hooks import FULL_TRACEBACK
    print(FULL_TRACEBACK)