import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def set_matlab_compatible(self):
    """ Sets options to return arrays as MATLAB loads them """
    self.mat_dtype = True
    self.squeeze_me = False
    self.chars_as_strings = False