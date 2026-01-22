import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def rlSkipIf(cond, reason, __module__=None):

    def inner(func):

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            if cond and os.environ.get('RL_indicateSkips', '0') == '1':
                print(f'\nskipping {func.__module__ or __module__}.{func.__name__} {reason}')
            return unittest.skipIf(cond, reason)(func)(*args, **kwds)
        return wrapper
    return inner