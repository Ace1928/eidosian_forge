import os
import py_compile
import marshal
import inspect
import re
import tokenize
from .command import Command
from . import pluginlib
def find_occurance(self, code):
    with self._open(code.co_filename) as f:
        lineno = 0
        for index, line in zip(range(code.co_firstlineno), f):
            lineno += 1
            pass
        first_indent = None
        for line in f:
            lineno += 1
            if line.find(self.symbol) != -1:
                this_indent = len(re.match('^[ \\t]*', line).group(0))
                if first_indent is None:
                    first_indent = this_indent
                elif this_indent < first_indent:
                    break
                print('  %3i  %s' % (lineno, line[first_indent:].rstrip()))
                if not self.verbose:
                    break