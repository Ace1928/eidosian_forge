from __future__ import absolute_import
import sys
from types import FunctionType
from future.utils import PY3, PY26
def superm(*args, **kwds):
    f = sys._getframe(1)
    nm = f.f_code.co_name
    return getattr(newsuper(framedepth=2), nm)(*args, **kwds)