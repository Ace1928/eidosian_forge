import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def test_strided(self):
    args_array = []
    for arg_idx in self.arguments:
        args_array.append(np.repeat(self.arguments[arg_idx][0], 2, axis=0)[::2])
    self.pythranfunc(*args_array)