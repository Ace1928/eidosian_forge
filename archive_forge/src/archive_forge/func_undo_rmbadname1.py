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
def undo_rmbadname1(name):
    if name in invbadnames:
        errmess('undo_rmbadname1: Replacing "%s" with "%s".\n' % (name, invbadnames[name]))
        return invbadnames[name]
    return name