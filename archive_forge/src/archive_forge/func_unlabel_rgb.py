import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def unlabel_rgb(colors):
    """
    Takes rgb color(s) 'rgb(a, b, c)' and returns tuple(s) (a, b, c)

    This function takes either an 'rgb(a, b, c)' color or a list of
    such colors and returns the color tuples in tuple(s) (a, b, c)
    """
    str_vals = ''
    for index in range(len(colors)):
        try:
            float(colors[index])
            str_vals = str_vals + colors[index]
        except ValueError:
            if colors[index] == ',' or colors[index] == '.':
                str_vals = str_vals + colors[index]
    str_vals = str_vals + ','
    numbers = []
    str_num = ''
    for char in str_vals:
        if char != ',':
            str_num = str_num + char
        else:
            numbers.append(float(str_num))
            str_num = ''
    return (numbers[0], numbers[1], numbers[2])