from ctypes import *
from ctypes.util import find_library
import os
def get_cc(self, chan, num):
    i = c_int()
    fluid_synth_get_cc(self.synth, chan, num, byref(i))
    return i.value