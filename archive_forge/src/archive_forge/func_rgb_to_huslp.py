import operator
import math
def rgb_to_huslp(r, g, b):
    return lch_to_huslp(rgb_to_lch(r, g, b))