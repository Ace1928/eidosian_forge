import numpy as np
from bokeh.core.properties import (
from ...core.options import abbreviated_exception
from ...core.util import arraylike_types
from ...util.transform import dim
from ..util import COLOR_ALIASES, RGB_HEX_REGEX, rgb2hex
def expand_batched_style(style, opts, mapping, nvals):
    """
    Computes styles applied to a batched plot by iterating over the
    supplied list of style options and expanding any options found in
    the supplied style dictionary returning a data and mapping defining
    the data that should be added to the ColumnDataSource.
    """
    opts = sorted(opts, key=lambda x: x in ['color', 'alpha'])
    applied_styles = set(mapping)
    style_data, style_mapping = ({}, {})
    for opt in opts:
        if 'color' in opt:
            alias = 'color'
        elif 'alpha' in opt:
            alias = 'alpha'
        else:
            alias = None
        if opt not in style or opt in mapping:
            continue
        elif opt == alias:
            if alias in applied_styles:
                continue
            elif 'line_' + alias in applied_styles:
                if 'fill_' + alias not in opts:
                    continue
                opt = 'fill_' + alias
                val = style[alias]
            elif 'fill_' + alias in applied_styles:
                opt = 'line_' + alias
                val = style[alias]
            else:
                val = style[alias]
        else:
            val = style[opt]
        style_mapping[opt] = {'field': opt}
        applied_styles.add(opt)
        if 'color' in opt and isinstance(val, tuple):
            val = rgb2hex(val)
        style_data[opt] = [val] * nvals
    return (style_data, style_mapping)