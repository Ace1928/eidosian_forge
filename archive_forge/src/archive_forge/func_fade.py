import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def fade(aSpotColor, percentages):
    """Waters down spot colors and returns a list of new ones

    e.g fade(myColor, [100,80,60,40,20]) returns a list of five colors
    """
    out = []
    for percent in percentages:
        frac = percent * 0.01
        newCyan = frac * aSpotColor.cyan
        newMagenta = frac * aSpotColor.magenta
        newYellow = frac * aSpotColor.yellow
        newBlack = frac * aSpotColor.black
        newDensity = frac * aSpotColor.density
        newSpot = CMYKColor(newCyan, newMagenta, newYellow, newBlack, spotName=aSpotColor.spotName, density=newDensity)
        out.append(newSpot)
    return out