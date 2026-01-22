import operator
import math
def hex_to_husl(hex):
    return rgb_to_husl(*hex_to_rgb(hex))