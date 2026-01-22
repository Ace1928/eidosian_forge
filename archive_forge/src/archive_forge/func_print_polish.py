from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def print_polish(self):
    """
        Print polish information
        """
    if self.work.first_run:
        runtime = self.work.info.setup_time + self.work.info.solve_time + self.work.info.polish_time
    else:
        runtime = self.work.info.update_time + self.work.info.solve_time + self.work.info.polish_time
    print('plsh  %11.4e   %8.2e   %8.2e   --------  %8.2es' % (self.work.info.obj_val, self.work.info.pri_res, self.work.info.dua_res, runtime))