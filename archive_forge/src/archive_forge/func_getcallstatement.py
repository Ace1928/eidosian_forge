import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getcallstatement(rout):
    return getmultilineblock(rout, 'callstatement')