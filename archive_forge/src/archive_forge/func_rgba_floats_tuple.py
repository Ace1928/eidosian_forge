import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
def rgba_floats_tuple(self, x):
    """
        Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with float values between 0. and 1.

        """
    if x <= self.index[0]:
        return self.colors[0]
    if x >= self.index[-1]:
        return self.colors[-1]
    i = len([u for u in self.index if u <= x])
    return tuple(self.colors[i - 1])