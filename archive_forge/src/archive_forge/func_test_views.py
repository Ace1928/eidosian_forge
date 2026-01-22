import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def test_views(self):
    args_array = []
    for arg_idx in self.arguments:
        args_array.append(self.arguments[arg_idx][0][::-1][::-1])
    self.pythranfunc(*args_array)