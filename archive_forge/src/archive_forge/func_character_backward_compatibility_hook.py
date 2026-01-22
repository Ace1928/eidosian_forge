import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def character_backward_compatibility_hook(item, parents, result, *args, **kwargs):
    """Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.

    See https://github.com/numpy/numpy/pull/19388 for more information.

    """
    parent_key, parent_value = parents[-1]
    key, value = item

    def fix_usage(varname, value):
        value = re.sub('[*]\\s*\\b' + varname + '\\b', varname, value)
        value = re.sub('\\b' + varname + '\\b\\s*[\\[]\\s*0\\s*[\\]]', varname, value)
        return value
    if parent_key in ['dimension', 'check']:
        assert parents[-3][0] == 'vars'
        vars_dict = parents[-3][1]
    elif key == '=':
        assert parents[-2][0] == 'vars'
        vars_dict = parents[-2][1]
    else:
        vars_dict = None
    new_value = None
    if vars_dict is not None:
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                new_value = fix_usage(varname, new_value)
    elif key == 'callstatement':
        vars_dict = parents[-2][1]['vars']
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                new_value = re.sub('(?<![&])\\b' + varname + '\\b', '&' + varname, new_value)
    if new_value is not None:
        if new_value != value:
            outmess(f'character_bc_hook[{parent_key}.{key}]: replaced `{value}` -> `{new_value}`\n', 1)
        return (key, new_value)