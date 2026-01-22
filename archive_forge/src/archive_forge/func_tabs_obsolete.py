from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def tabs_obsolete(physical_line):
    """For new projects, spaces-only are strongly recommended over tabs.

    Okay: if True:\\n    return
    W191: if True:\\n\\treturn
    """
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return (indent.index('\t'), 'W191 indentation contains tabs')