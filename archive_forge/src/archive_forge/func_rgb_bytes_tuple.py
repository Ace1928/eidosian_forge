import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def rgb_bytes_tuple(self, x):
    """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B) with int values between 0 and 255.
        """
    return self.rgba_bytes_tuple(x)[:3]