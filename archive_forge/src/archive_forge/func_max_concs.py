from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
def max_concs(self, params, min_=min, dtype=np.float64):
    init_concs = params[:self.eqsys.ns]
    return self.eqsys.upper_conc_bounds(init_concs, min_=min_, dtype=dtype)