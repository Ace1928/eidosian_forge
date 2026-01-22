import math
import warnings
import matplotlib.dates
def get_rect_ymin(data):
    """Find minimum y value from four (x,y) vertices."""
    return min(data[0][1], data[1][1], data[2][1], data[3][1])