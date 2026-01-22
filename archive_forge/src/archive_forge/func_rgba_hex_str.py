import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def rgba_hex_str(self, x):
    """Provides the color corresponding to value `x` in the
        form of a string of hexadecimal values "#RRGGBBAA".
        """
    return '#%02x%02x%02x%02x' % self.rgba_bytes_tuple(x)