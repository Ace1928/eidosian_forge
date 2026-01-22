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
def fix_usage(varname, value):
    value = re.sub('[*]\\s*\\b' + varname + '\\b', varname, value)
    value = re.sub('\\b' + varname + '\\b\\s*[\\[]\\s*0\\s*[\\]]', varname, value)
    return value