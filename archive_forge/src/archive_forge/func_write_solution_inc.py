from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_solution_inc(f, data):
    """
    Prototypes for solution vectors
    """
    f.write('// Prototypes for solution\n')
    f.write('extern c_float xsolution[%d];\n' % data['n'])
    f.write('extern c_float ysolution[%d];\n\n' % data['m'])
    f.write('extern OSQPSolution solution;\n\n')