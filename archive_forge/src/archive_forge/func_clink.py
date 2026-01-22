from __future__ import absolute_import
import sys
import os
def clink(basename):
    runcmd([LINKCC, '-o', basename + EXE_EXT, basename + '.o', '-L' + LIBDIR1, '-L' + LIBDIR2] + [PYLIB_DYN and '-l' + PYLIB_DYN or os.path.join(LIBDIR1, PYLIB)] + LIBS.split() + SYSLIBS.split() + LINKFORSHARED.split())