import cmath
import numpy as np
from numba import float32
from numba.types import unicode_type, i8
from numba.pycc import CC, exportmany, export
from numba.tests.support import has_blas
from numba import typed
@cc_nrt.export('hash_literal_str_A', i8())
def internal_str_dict():
    return hash('A')