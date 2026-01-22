import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def recurser(index, hanging_indent, curr_width):
    """
        By using this local function, we don't need to recurse with all the
        arguments. Since this function is not created recursively, the cost is
        not significant
        """
    axis = len(index)
    axes_left = a.ndim - axis
    if axes_left == 0:
        return format_function(a[index])
    next_hanging_indent = hanging_indent + ' '
    if legacy <= 113:
        next_width = curr_width
    else:
        next_width = curr_width - len(']')
    a_len = a.shape[axis]
    show_summary = summary_insert and 2 * edge_items < a_len
    if show_summary:
        leading_items = edge_items
        trailing_items = edge_items
    else:
        leading_items = 0
        trailing_items = a_len
    s = ''
    if axes_left == 1:
        if legacy <= 113:
            elem_width = curr_width - len(separator.rstrip())
        else:
            elem_width = curr_width - max(len(separator.rstrip()), len(']'))
        line = hanging_indent
        for i in range(leading_items):
            word = recurser(index + (i,), next_hanging_indent, next_width)
            s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
            line += separator
        if show_summary:
            s, line = _extendLine(s, line, summary_insert, elem_width, hanging_indent, legacy)
            if legacy <= 113:
                line += ', '
            else:
                line += separator
        for i in range(trailing_items, 1, -1):
            word = recurser(index + (-i,), next_hanging_indent, next_width)
            s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
            line += separator
        if legacy <= 113:
            elem_width = curr_width
        word = recurser(index + (-1,), next_hanging_indent, next_width)
        s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
        s += line
    else:
        s = ''
        line_sep = separator.rstrip() + '\n' * (axes_left - 1)
        for i in range(leading_items):
            nested = recurser(index + (i,), next_hanging_indent, next_width)
            s += hanging_indent + nested + line_sep
        if show_summary:
            if legacy <= 113:
                s += hanging_indent + summary_insert + ', \n'
            else:
                s += hanging_indent + summary_insert + line_sep
        for i in range(trailing_items, 1, -1):
            nested = recurser(index + (-i,), next_hanging_indent, next_width)
            s += hanging_indent + nested + line_sep
        nested = recurser(index + (-1,), next_hanging_indent, next_width)
        s += hanging_indent + nested
    s = '[' + s[len(hanging_indent):] + ']'
    return s