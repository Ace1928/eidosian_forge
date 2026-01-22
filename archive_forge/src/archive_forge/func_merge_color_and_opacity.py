import math
import warnings
import matplotlib.dates
def merge_color_and_opacity(color, opacity):
    """
    Merge hex color with an alpha (opacity) to get an rgba tuple.

    :param (str|unicode) color: A hex color string.
    :param (float|int) opacity: A value [0, 1] for the 'a' in 'rgba'.
    :return: (int, int, int, float) The rgba color and alpha tuple.

    """
    if color is None:
        return None
    rgb_tup = hex_to_rgb(color)
    if opacity is None:
        return 'rgb {}'.format(rgb_tup)
    rgba_tup = rgb_tup + (opacity,)
    return 'rgba {}'.format(rgba_tup)