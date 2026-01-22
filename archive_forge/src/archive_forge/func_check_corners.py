import math
import warnings
import matplotlib.dates
def check_corners(inner_obj, outer_obj):
    inner_corners = inner_obj.get_window_extent().corners()
    outer_corners = outer_obj.get_window_extent().corners()
    if inner_corners[0][0] < outer_corners[0][0]:
        return False
    elif inner_corners[0][1] < outer_corners[0][1]:
        return False
    elif inner_corners[3][0] > outer_corners[3][0]:
        return False
    elif inner_corners[3][1] > outer_corners[3][1]:
        return False
    else:
        return True