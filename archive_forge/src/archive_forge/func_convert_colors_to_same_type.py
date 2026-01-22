import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def convert_colors_to_same_type(colors, colortype='rgb', scale=None, return_default_colors=False, num_of_defualt_colors=2):
    """
    Converts color(s) to the specified color type

    Takes a single color or an iterable of colors, as well as a list of scale
    values, and outputs a 2-pair of the list of color(s) converted all to an
    rgb or tuple color type, aswell as the scale as the second element. If
    colors is a Plotly Scale name, then 'scale' will be forced to the scale
    from the respective colorscale and the colors in that colorscale will also
    be coverted to the selected colortype. If colors is None, then there is an
    option to return portion of the DEFAULT_PLOTLY_COLORS

    :param (str|tuple|list) colors: either a plotly scale name, an rgb or hex
        color, a color tuple or a list/tuple of colors
    :param (list) scale: see docs for validate_scale_values()

    :rtype (tuple) (colors_list, scale) if scale is None in the function call,
        then scale will remain None in the returned tuple
    """
    colors_list = []
    if colors is None and return_default_colors is True:
        colors_list = DEFAULT_PLOTLY_COLORS[0:num_of_defualt_colors]
    if isinstance(colors, str):
        if colors in PLOTLY_SCALES:
            colors_list = colorscale_to_colors(PLOTLY_SCALES[colors])
            if scale is None:
                scale = colorscale_to_scale(PLOTLY_SCALES[colors])
        elif 'rgb' in colors or '#' in colors:
            colors_list = [colors]
    elif isinstance(colors, tuple):
        if isinstance(colors[0], Number):
            colors_list = [colors]
        else:
            colors_list = list(colors)
    elif isinstance(colors, list):
        colors_list = colors
    if scale is not None:
        validate_scale_values(scale)
        if len(colors_list) != len(scale):
            raise exceptions.PlotlyError('Make sure that the length of your scale matches the length of your list of colors which is {}.'.format(len(colors_list)))
    for j, each_color in enumerate(colors_list):
        if '#' in each_color:
            each_color = color_parser(each_color, hex_to_rgb)
            each_color = color_parser(each_color, label_rgb)
            colors_list[j] = each_color
        elif isinstance(each_color, tuple):
            each_color = color_parser(each_color, convert_to_RGB_255)
            each_color = color_parser(each_color, label_rgb)
            colors_list[j] = each_color
    if colortype == 'rgb':
        return (colors_list, scale)
    elif colortype == 'tuple':
        for j, each_color in enumerate(colors_list):
            each_color = color_parser(each_color, unlabel_rgb)
            each_color = color_parser(each_color, unconvert_from_RGB_255)
            colors_list[j] = each_color
        return (colors_list, scale)
    else:
        raise exceptions.PlotlyError('You must select either rgb or tuple for your colortype variable.')