from __future__ import absolute_import
import sys
import os
def ccompile(basename):
    runcmd([CC, '-c', '-o', basename + '.o', basename + '.c', '-I' + INCDIR] + CFLAGS.split())