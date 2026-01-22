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
def true_intent_list(var):
    lst = var['intent']
    ret = []
    for intent in lst:
        try:
            f = globals()['isintent_%s' % intent]
        except KeyError:
            pass
        else:
            if f(var):
                ret.append(intent)
    return ret