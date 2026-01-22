from ctypes import *
from ctypes.util import find_library
import os
def program_change(self, chan, prg):
    """Change the program"""
    return fluid_synth_program_change(self.synth, chan, prg)