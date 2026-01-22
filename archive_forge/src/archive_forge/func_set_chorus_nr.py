from ctypes import *
from ctypes.util import find_library
import os
def set_chorus_nr(self, nr):
    if fluid_synth_set_chorus_nr is not None:
        return fluid_synth_set_chorus_nr(self.synth, nr)
    else:
        return self.set_chorus(nr=nr)