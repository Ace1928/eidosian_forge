import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isstring_or_stringarray(var):
    return _ischaracter(var) and 'charselector' in var