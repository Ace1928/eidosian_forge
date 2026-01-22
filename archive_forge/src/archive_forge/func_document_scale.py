from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def document_scale(cls: type[scale]) -> type[scale]:
    """
    Create a documentation for a scale

    Import the superclass parameters

    It replaces `{superclass_parameters}` with the documentation
    of the parameters from the superclass.

    Parameters
    ----------
    cls : type
        A scale class

    Returns
    -------
    cls : type
        The scale class with a modified docstring.
    """
    params_list = []
    cls_param_string = docstring_parameters_section(cls)
    cls_param_dict = parameters_str_to_dict(cls_param_string)
    cls_params = set(cls_param_dict.keys())
    for i, base in enumerate(cls.__bases__):
        base_param_string = param_string = docstring_parameters_section(base)
        base_param_dict = parameters_str_to_dict(base_param_string)
        base_params = set(base_param_dict.keys())
        duplicate_params = base_params & cls_params
        for param in duplicate_params:
            del base_param_dict[param]
        if duplicate_params:
            param_string = parameters_dict_to_str(base_param_dict)
        if i == 0:
            param_string = param_string.strip()
        params_list.append(param_string)
        cls_params |= base_params
    superclass_parameters = '\n'.join(params_list)
    cls_doc = cls.__doc__ or ''
    cls.__doc__ = cls_doc.format(superclass_parameters=superclass_parameters)
    return cls