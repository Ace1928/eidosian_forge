import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def hasexternals(rout):
    return 'externals' in rout and rout['externals']