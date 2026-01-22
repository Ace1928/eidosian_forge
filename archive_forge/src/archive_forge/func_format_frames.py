import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def format_frames(frames, write, detail='default'):
    """
    Format and write frames.

    @param frames: is a list of frames as used by Failure.frames, with
        each frame being a list of
        (funcName, fileName, lineNumber, locals.items(), globals.items())
    @type frames: list
    @param write: this will be called with formatted strings.
    @type write: callable
    @param detail: Four detail levels are available:
        default, brief, verbose, and verbose-vars-not-captured.
        C{Failure.printDetailedTraceback} uses the latter when the caller asks
        for verbose, but no vars were captured, so that an explicit warning
        about the missing data is shown.
    @type detail: string
    """
    if detail not in ('default', 'brief', 'verbose', 'verbose-vars-not-captured'):
        raise ValueError('Detail must be default, brief, verbose, or verbose-vars-not-captured. (not %r)' % (detail,))
    w = write
    if detail == 'brief':
        for method, filename, lineno, localVars, globalVars in frames:
            w(f'{filename}:{lineno}:{method}\n')
    elif detail == 'default':
        for method, filename, lineno, localVars, globalVars in frames:
            w(f'  File "{filename}", line {lineno}, in {method}\n')
            w('    %s\n' % linecache.getline(filename, lineno).strip())
    elif detail == 'verbose-vars-not-captured':
        for method, filename, lineno, localVars, globalVars in frames:
            w('%s:%d: %s(...)\n' % (filename, lineno, method))
        w(' [Capture of Locals and Globals disabled (use captureVars=True)]\n')
    elif detail == 'verbose':
        for method, filename, lineno, localVars, globalVars in frames:
            w('%s:%d: %s(...)\n' % (filename, lineno, method))
            w(' [ Locals ]\n')
            for name, val in localVars:
                w(f'  {name} : {repr(val)}\n')
            w(' ( Globals )\n')
            for name, val in globalVars:
                w(f'  {name} : {repr(val)}\n')