import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
@property
def tmpName(self):
    name = '%s%s' % (self._tmp_pfx, self._tmp_idx)
    self._tmp_idx += 1
    return name