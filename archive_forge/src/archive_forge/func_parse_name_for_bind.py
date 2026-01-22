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
def parse_name_for_bind(line):
    pattern = re.compile('bind\\(\\s*(?P<lang>[^,]+)(?:\\s*,\\s*name\\s*=\\s*["\\\'](?P<name>[^"\\\']+)["\\\']\\s*)?\\)', re.I)
    match = pattern.search(line)
    bind_statement = None
    if match:
        bind_statement = match.group(0)
        line = line[:match.start()] + line[match.end():]
    return (line, bind_statement)