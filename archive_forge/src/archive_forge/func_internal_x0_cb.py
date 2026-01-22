from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
def internal_x0_cb(self, init_concs, params):
    return [0.1] * len(init_concs)