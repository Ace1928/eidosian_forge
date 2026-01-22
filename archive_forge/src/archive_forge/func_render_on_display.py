import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict
@render_on_display.setter
def render_on_display(self, val):
    self._render_on_display = bool(val)