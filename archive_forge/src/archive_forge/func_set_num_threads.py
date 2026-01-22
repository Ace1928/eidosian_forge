import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def set_num_threads(self, num_threads):
    global PARALLELISM
    PARALLELISM = num_threads