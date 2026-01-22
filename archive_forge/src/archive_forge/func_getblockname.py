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
def getblockname(block, unknown='unknown'):
    if 'name' in block:
        return block['name']
    return unknown