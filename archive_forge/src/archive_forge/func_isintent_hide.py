import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isintent_hide(var):
    return 'intent' in var and ('hide' in var['intent'] or ('out' in var['intent'] and 'in' not in var['intent'] and (not l_or(isintent_inout, isintent_inplace)(var))))