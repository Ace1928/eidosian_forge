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
def sortvarnames(vars):
    indep = []
    dep = []
    for v in list(vars.keys()):
        if 'depend' in vars[v] and vars[v]['depend']:
            dep.append(v)
        else:
            indep.append(v)
    n = len(dep)
    i = 0
    while dep:
        v = dep[0]
        fl = 0
        for w in dep[1:]:
            if w in vars[v]['depend']:
                fl = 1
                break
        if fl:
            dep = dep[1:] + [v]
            i = i + 1
            if i > n:
                errmess('sortvarnames: failed to compute dependencies because of cyclic dependencies between ' + ', '.join(dep) + '\n')
                indep = indep + dep
                break
        else:
            indep.append(v)
            dep = dep[1:]
            n = len(dep)
            i = 0
    return indep