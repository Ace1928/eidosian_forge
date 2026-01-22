import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setParseAction(self, *fns, **kwargs):
    """Define action to perform when successfully matching parse element definition.
           Parse action fn is a callable method with 0-3 arguments, called as C{fn(s,loc,toks)},
           C{fn(loc,toks)}, C{fn(toks)}, or just C{fn()}, where:
            - s   = the original string being parsed (see note below)
            - loc = the location of the matching substring
            - toks = a list of the matched tokens, packaged as a C{L{ParseResults}} object
           If the functions in fns modify the tokens, they can return them as the return
           value from fn, and the modified list of tokens will replace the original.
           Otherwise, fn does not need to return any value.

           Note: the default parsing behavior is to expand tabs in the input string
           before starting the parsing process.  See L{I{parseString}<parseString>} for more information
           on parsing strings containing C{<TAB>}s, and suggested methods to maintain a
           consistent view of the parsed string, the parse location, and line and column
           positions within the parsed string.
           """
    self.parseAction = list(map(_trim_arity, list(fns)))
    self.callDuringTry = 'callDuringTry' in kwargs and kwargs['callDuringTry']
    return self