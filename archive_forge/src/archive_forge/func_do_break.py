import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_break(self, arg, temporary=0):
    """b(reak) [ ([filename:]lineno | function) [, condition] ]
        Without argument, list all breaks.

        With a line number argument, set a break at this line in the
        current file.  With a function name, set a break at the first
        executable line of that function.  If a second argument is
        present, it is a string specifying an expression which must
        evaluate to true before the breakpoint is honored.

        The line number may be prefixed with a filename and a colon,
        to specify a breakpoint in another file (probably one that
        hasn't been loaded yet).  The file is searched for on
        sys.path; the .py suffix may be omitted.
        """
    if not arg:
        if self.breaks:
            self.message('Num Type         Disp Enb   Where')
            for bp in bdb.Breakpoint.bpbynumber:
                if bp:
                    self.message(bp.bpformat())
        return
    filename = None
    lineno = None
    cond = None
    comma = arg.find(',')
    if comma > 0:
        cond = arg[comma + 1:].lstrip()
        arg = arg[:comma].rstrip()
    colon = arg.rfind(':')
    funcname = None
    if colon >= 0:
        filename = arg[:colon].rstrip()
        f = self.lookupmodule(filename)
        if not f:
            self.error('%r not found from sys.path' % filename)
            return
        else:
            filename = f
        arg = arg[colon + 1:].lstrip()
        try:
            lineno = int(arg)
        except ValueError:
            self.error('Bad lineno: %s' % arg)
            return
    else:
        try:
            lineno = int(arg)
        except ValueError:
            try:
                func = eval(arg, self.curframe.f_globals, self.curframe_locals)
            except:
                func = arg
            try:
                if hasattr(func, '__func__'):
                    func = func.__func__
                code = func.__code__
                funcname = code.co_name
                lineno = code.co_firstlineno
                filename = code.co_filename
            except:
                ok, filename, ln = self.lineinfo(arg)
                if not ok:
                    self.error('The specified object %r is not a function or was not found along sys.path.' % arg)
                    return
                funcname = ok
                lineno = int(ln)
    if not filename:
        filename = self.defaultFile()
    line = self.checkline(filename, lineno)
    if line:
        err = self.set_break(filename, line, temporary, cond, funcname)
        if err:
            self.error(err)
        else:
            bp = self.get_breaks(filename, line)[-1]
            self.message('Breakpoint %d at %s:%d' % (bp.number, bp.file, bp.line))