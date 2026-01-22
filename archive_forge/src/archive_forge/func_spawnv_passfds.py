import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def spawnv_passfds(path, args, passfds):
    import _posixsubprocess
    passfds = tuple(sorted(map(int, passfds)))
    errpipe_read, errpipe_write = os.pipe()
    try:
        return _posixsubprocess.fork_exec(args, [os.fsencode(path)], True, passfds, None, None, -1, -1, -1, -1, -1, -1, errpipe_read, errpipe_write, False, False, None, None, None, -1, None)
    finally:
        os.close(errpipe_read)
        os.close(errpipe_write)