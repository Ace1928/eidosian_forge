import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isintent_inout(var):
    return 'intent' in var and ('inout' in var['intent'] or 'outin' in var['intent']) and ('in' not in var['intent']) and ('hide' not in var['intent']) and ('inplace' not in var['intent'])