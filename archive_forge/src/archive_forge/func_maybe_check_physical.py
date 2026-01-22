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
def maybe_check_physical(self, token):
    """If appropriate (based on token), check current physical line(s)."""
    if _is_eol_token(token):
        self.check_physical(token[4])
    elif token[0] == tokenize.STRING and '\n' in token[1]:
        if noqa(token[4]):
            return
        self.multiline = True
        self.line_number = token[2][0]
        for line in token[1].split('\n')[:-1]:
            self.check_physical(line + '\n')
            self.line_number += 1
        self.multiline = False