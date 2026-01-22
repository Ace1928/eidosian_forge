from __future__ import generators
from bisect import bisect_right
import sys
import inspect, tokenize
import py
from types import ModuleType
def isparseable(self, deindent=True):
    """ return True if source is parseable, heuristically
            deindenting it by default.
        """
    try:
        import parser
    except ImportError:
        syntax_checker = lambda x: compile(x, 'asd', 'exec')
    else:
        syntax_checker = parser.suite
    if deindent:
        source = str(self.deindent())
    else:
        source = str(self)
    try:
        syntax_checker(source + '\n')
    except KeyboardInterrupt:
        raise
    except Exception:
        return False
    else:
        return True