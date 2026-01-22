from ctypes import *
from ctypes.util import find_library
import os
def program_select(self, chan, sfid, bank, preset):
    """Select a program"""
    return fluid_synth_program_select(self.synth, chan, sfid, bank, preset)