import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def rgba_bytes_tuple(self, x):
    """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with int values between 0 and 255.
        """
    return tuple((int(u * 255.9999) for u in self.rgba_floats_tuple(x)))