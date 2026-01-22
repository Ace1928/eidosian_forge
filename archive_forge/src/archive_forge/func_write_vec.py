from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_vec(f, vec, name, vec_type):
    """
    Write vector to file
    """
    if len(vec) > 0:
        f.write('%s %s[%d] = {\n' % (vec_type, name, len(vec)))
        for i in range(len(vec)):
            if vec_type == 'c_float':
                f.write('(c_float)%.20f,\n' % vec[i])
            else:
                f.write('%i,\n' % vec[i])
        f.write('};\n')