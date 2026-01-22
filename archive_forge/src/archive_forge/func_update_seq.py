import numbers
from functools import reduce
from operator import mul
import numpy as np
def update_seq(self, arr_seq):
    arr_seq._offsets = np.array(self.offsets)
    arr_seq._lengths = np.array(self.lengths)