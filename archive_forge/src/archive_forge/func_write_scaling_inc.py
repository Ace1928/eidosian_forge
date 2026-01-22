from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_scaling_inc(f, scaling):
    """
    Write prototypes for the scaling structure to file
    """
    f.write('// Scaling structure prototypes\n')
    if scaling is not None:
        write_vec_extern(f, scaling['D'], 'Dscaling', 'c_float')
        write_vec_extern(f, scaling['Dinv'], 'Dinvscaling', 'c_float')
        write_vec_extern(f, scaling['E'], 'Escaling', 'c_float')
        write_vec_extern(f, scaling['Einv'], 'Einvscaling', 'c_float')
    f.write('extern OSQPScaling scaling;\n\n')