from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
def debug_decorator(func):
    """If SYMPY_DEBUG is True, it will print a nice execution tree with
    arguments and results of all decorated functions, else do nothing.
    """
    from sympy import SYMPY_DEBUG
    if not SYMPY_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp
        global _debug_iter
        oldtmp = _debug_tmp
        _debug_tmp = []
        _debug_iter += 1

        def tree(subtrees):

            def indent(s, variant=1):
                x = s.split('\n')
                r = '+-%s\n' % x[0]
                for a in x[1:]:
                    if a == '':
                        continue
                    if variant == 1:
                        r += '| %s\n' % a
                    else:
                        r += '  %s\n' % a
                return r
            if len(subtrees) == 0:
                return ''
            f = []
            for a in subtrees[:-1]:
                f.append(indent(a))
            f.append(indent(subtrees[-1], 2))
            return ''.join(f)
        r = f(*args, **kw)
        _debug_iter -= 1
        s = '%s%s = %s\n' % (f.__name__, args, r)
        if _debug_tmp != []:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        _debug_tmp.append(s)
        if _debug_iter == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)
    return decorated