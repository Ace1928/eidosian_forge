from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_mat(f, mat, name):
    """
    Write scipy sparse matrix in CSC form to file
    """
    write_vec(f, mat['p'], name + '_p', 'c_int')
    if len(mat['x']) > 0:
        write_vec(f, mat['i'], name + '_i', 'c_int')
        write_vec(f, mat['x'], name + '_x', 'c_float')
    f.write('csc %s = {' % name)
    f.write('%d, ' % mat['nzmax'])
    f.write('%d, ' % mat['m'])
    f.write('%d, ' % mat['n'])
    f.write('%s_p, ' % name)
    if len(mat['x']) > 0:
        f.write('%s_i, ' % name)
        f.write('%s_x, ' % name)
    else:
        f.write('0, 0, ')
    f.write('%d};\n' % mat['nz'])