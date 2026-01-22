import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isoptional(var):
    return ('attrspec' in var and 'optional' in var['attrspec'] and ('required' not in var['attrspec'])) and isintent_nothide(var)