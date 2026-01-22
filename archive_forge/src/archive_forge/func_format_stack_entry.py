import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def format_stack_entry(self, frame_lineno, lprefix=': '):
    """Return a string with information about a stack entry.

        The stack entry frame_lineno is a (frame, lineno) tuple.  The
        return string contains the canonical filename, the function name
        or '<lambda>', the input arguments, the return value, and the
        line of code (if it exists).

        """
    import linecache, reprlib
    frame, lineno = frame_lineno
    filename = self.canonic(frame.f_code.co_filename)
    s = '%s(%r)' % (filename, lineno)
    if frame.f_code.co_name:
        s += frame.f_code.co_name
    else:
        s += '<lambda>'
    s += '()'
    if '__return__' in frame.f_locals:
        rv = frame.f_locals['__return__']
        s += '->'
        s += reprlib.repr(rv)
    if lineno is not None:
        line = linecache.getline(filename, lineno, frame.f_globals)
        if line:
            s += lprefix + line.strip()
    else:
        s += f'{lprefix}Warning: lineno is None'
    return s