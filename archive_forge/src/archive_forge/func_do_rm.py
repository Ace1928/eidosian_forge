import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def do_rm(self, test_case, input, args):
    err = None

    def error(msg, path):
        return "rm: cannot remove '{}': {}\n".format(path, msg)
    force, recursive = (False, False)
    opts = None
    if args and args[0][0] == '-':
        opts = args.pop(0)[1:]
        if 'f' in opts:
            force = True
            opts = opts.replace('f', '', 1)
        if 'r' in opts:
            recursive = True
            opts = opts.replace('r', '', 1)
    if not args or opts:
        raise SyntaxError('Usage: rm [-fr] path+')
    for p in args:
        self._ensure_in_jail(test_case, p)
        try:
            os.remove(p)
        except OSError as e:
            if e.errno in (errno.EISDIR, errno.EPERM, errno.EACCES):
                if recursive:
                    osutils.rmtree(p)
                else:
                    err = error('Is a directory', p)
                    break
            elif e.errno == errno.ENOENT:
                if not force:
                    err = error('No such file or directory', p)
                    break
            else:
                raise
    if err:
        retcode = 1
    else:
        retcode = 0
    return (retcode, None, err)