from collections import namedtuple
from math import floor, ceil
def draw_two_pixel(float_x, in_y, factor):
    fl_x = floor(float_x)
    draw_pixel(surf, (fl_x, in_y), color, factor * inv_frac(float_x), blend)
    draw_pixel(surf, (fl_x + 1, in_y), color, factor * frac(float_x), blend)