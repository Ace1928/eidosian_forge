from ctypes import *
from ctypes.util import find_library
import os
def program_reset(self):
    """Reset the programs on all channels"""
    return fluid_synth_program_reset(self.synth)