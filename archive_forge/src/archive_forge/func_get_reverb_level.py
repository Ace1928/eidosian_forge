from ctypes import *
from ctypes.util import find_library
import os
def get_reverb_level(self):
    return fluid_synth_get_reverb_level(self.synth)