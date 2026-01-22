import sys
import os
import pprint
import re
from pathlib import Path
from itertools import dropwhile
import argparse
import copy
from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
from numpy.f2py._backends import f2py_build_generator
def validate_modulename(pyf_files, modulename='untitled'):
    if len(pyf_files) > 1:
        raise ValueError('Only one .pyf file per call')
    if pyf_files:
        pyff = pyf_files[0]
        pyf_modname = auxfuncs.get_f2py_modulename(pyff)
        if modulename != pyf_modname:
            outmess(f'Ignoring -m {modulename}.\n{pyff} defines {pyf_modname} to be the modulename.\n')
            modulename = pyf_modname
    return modulename