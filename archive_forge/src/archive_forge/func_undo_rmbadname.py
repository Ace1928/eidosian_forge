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
def undo_rmbadname(names):
    return [undo_rmbadname1(_m) for _m in names]