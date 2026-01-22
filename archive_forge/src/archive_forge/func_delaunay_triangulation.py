from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
def delaunay_triangulation(self, n_prc=0):
    if hasattr(self, 'Tri') and self.qhull_incremental:
        self.Tri.add_points(self.C[n_prc:, :])
    else:
        try:
            self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
        except spatial.QhullError:
            if str(sys.exc_info()[1])[:6] == 'QH6239':
                logging.warning('QH6239 Qhull precision error detected, this usually occurs when no bounds are specified, Qhull can only run with handling cocircular/cospherical points and in this case incremental mode is switched off. The performance of shgo will be reduced in this mode.')
                self.qhull_incremental = False
                self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
            else:
                raise
    return self.Tri