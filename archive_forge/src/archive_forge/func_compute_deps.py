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
def compute_deps(v, deps):
    for v1 in coeffs_and_deps.get(v, [None, []])[1]:
        if v1 not in deps:
            deps.add(v1)
            compute_deps(v1, deps)