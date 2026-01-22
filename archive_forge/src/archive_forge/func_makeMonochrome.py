from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def makeMonochrome(color='neutral'):
    """
    Returns a ColorMap object with a dark to bright ramp and adjustable tint.
    
    In addition to neutral, warm or cold grays, imitations of monochrome computer monitors are also
    available. The following predefined color ramps are available:
    `neutral`, `warm`, `cool`, `green`, `amber`, `blue`, `red`, `pink`, `lavender`.
    
    The ramp can also be specified by a tuple of float values in the range of 0 to 1.
    In this case `(h, s, l0, l1)` describe hue, saturation, minimum lightness and maximum lightness
    within the HSL color space. The values `l0` and `l1` can be omitted. They default to 
    `l0=0.0` and `l1=1.0` in this case.

    Parameters
    ----------
    color: str or tuple of floats
        Color description. Can be one of the predefined identifiers, or a tuple
        `(h, s, l0, l1)`, `(h, s)` or (`h`).
        'green', 'amber', 'blue', 'red', 'lavender', 'pink'
        or a tuple of relative ``(R,G,B)`` contributions in range 0.0 to 1.0
    """
    name = f'Monochrome {color}'
    defaults = {'neutral': (0.0, 0.0, 0.0, 1.0), 'warm': (0.1, 0.08, 0.0, 0.95), 'cool': (0.6, 0.08, 0.0, 0.95), 'green': (0.35, 0.55, 0.02, 0.9), 'amber': (0.09, 0.8, 0.02, 0.8), 'blue': (0.58, 0.85, 0.02, 0.95), 'red': (0.01, 0.6, 0.02, 0.9), 'pink': (0.93, 0.65, 0.02, 0.95), 'lavender': (0.75, 0.5, 0.02, 0.9)}
    if isinstance(color, str):
        if color in defaults:
            h_val, s_val, l_min, l_max = defaults[color]
        else:
            valid = ','.join(defaults.keys())
            raise ValueError(f"Undefined color descriptor '{color}', known values are:\n{valid}")
    else:
        s_val = 0.7
        l_min = 0.0
        l_max = 1.0
        if not hasattr(color, '__len__'):
            h_val = float(color)
        elif len(color) == 1:
            h_val = color[0]
        elif len(color) == 2:
            h_val, s_val = color
        elif len(color) == 4:
            h_val, s_val, l_min, l_max = color
        else:
            raise ValueError(f"Invalid color descriptor '{color}'")
    l_vals = np.linspace(l_min, l_max, num=16)
    color_list = []
    for l_val in l_vals:
        qcol = QtGui.QColor.fromHslF(h_val, s_val, l_val)
        color_list.append(qcol)
    return ColorMap(None, color_list, name=name, linearize=True)