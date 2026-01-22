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
def get_useparameters(block, param_map=None):
    global f90modulevars
    if param_map is None:
        param_map = {}
    usedict = get_usedict(block)
    if not usedict:
        return param_map
    for usename, mapping in list(usedict.items()):
        usename = usename.lower()
        if usename not in f90modulevars:
            outmess('get_useparameters: no module %s info used by %s\n' % (usename, block.get('name')))
            continue
        mvars = f90modulevars[usename]
        params = get_parameters(mvars)
        if not params:
            continue
        if mapping:
            errmess('get_useparameters: mapping for %s not impl.\n' % mapping)
        for k, v in list(params.items()):
            if k in param_map:
                outmess('get_useparameters: overriding parameter %s with value from module %s\n' % (repr(k), repr(usename)))
            param_map[k] = v
    return param_map