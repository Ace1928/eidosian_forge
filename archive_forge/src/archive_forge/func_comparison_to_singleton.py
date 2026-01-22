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
def comparison_to_singleton(logical_line, noqa):
    """Comparison to singletons should use "is" or "is not".

    Comparisons to singletons like None should always be done
    with "is" or "is not", never the equality operators.

    Okay: if arg is not None:
    E711: if arg != None:
    E711: if None == arg:
    E712: if arg == True:
    E712: if False == arg:

    Also, beware of writing if x when you really mean if x is not None --
    e.g. when testing whether a variable or argument that defaults to None was
    set to some other value.  The other value might have a type (such as a
    container) that could be false in a boolean context!
    """
    match = not noqa and COMPARE_SINGLETON_REGEX.search(logical_line)
    if match:
        singleton = match.group(1) or match.group(3)
        same = match.group(2) == '=='
        msg = "'if cond is %s:'" % (('' if same else 'not ') + singleton)
        if singleton in ('None',):
            code = 'E711'
        else:
            code = 'E712'
            nonzero = singleton == 'True' and same or (singleton == 'False' and (not same))
            msg += " or 'if %scond:'" % ('' if nonzero else 'not ')
        yield (match.start(2), '%s comparison to %s should be %s' % (code, singleton, msg))