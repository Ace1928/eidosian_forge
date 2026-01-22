from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_vec_extern(f, vec, name, vec_type):
    """
    Write vector prototype to file
    """
    if len(vec) > 0:
        f.write('extern %s %s[%d];\n' % (vec_type, name, len(vec)))