from ctypes import *
from ctypes.util import find_library
import os
def set_chorus_type(self, type):
    if fluid_synth_set_chorus_type is not None:
        return fluid_synth_set_chorus_type(self.synth, type)
    else:
        return self.set_chorus(type=type)