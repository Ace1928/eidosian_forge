import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def outmess(t):
    if options.get('verbose', 1):
        sys.stdout.write(t)