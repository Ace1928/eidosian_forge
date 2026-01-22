import math
import warnings
import matplotlib.dates
def convert_linewidth_array(width_array):
    if len(width_array) == 1:
        return width_array[0]
    else:
        return width_array