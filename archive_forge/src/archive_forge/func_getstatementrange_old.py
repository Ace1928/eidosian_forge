from __future__ import generators
from bisect import bisect_right
import sys
import inspect, tokenize
import py
from types import ModuleType
def getstatementrange_old(lineno, source, assertion=False):
    """ return (start, end) tuple which spans the minimal
        statement region which containing the given lineno.
        raise an IndexError if no such statementrange can be found.
    """
    from codeop import compile_command
    for start in range(lineno, -1, -1):
        if assertion:
            line = source.lines[start]
            if 'super' in line and 'self' in line and ('__init__' in line):
                raise IndexError('likely a subclass')
            if 'assert' not in line and 'raise' not in line:
                continue
        trylines = source.lines[start:lineno + 1]
        trylines.insert(0, 'def xxx():')
        trysource = '\n '.join(trylines)
        try:
            compile_command(trysource)
        except (SyntaxError, OverflowError, ValueError):
            continue
        for end in range(lineno + 1, len(source) + 1):
            trysource = source[start:end]
            if trysource.isparseable():
                return (start, end)
    raise SyntaxError('no valid source range around line %d ' % (lineno,))