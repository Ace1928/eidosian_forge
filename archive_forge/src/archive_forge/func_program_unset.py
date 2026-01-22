from ctypes import *
from ctypes.util import find_library
import os
def program_unset(self, chan):
    """Set the preset of a MIDI channel to an unassigned state"""
    return fluid_synth_unset_program(self.synth, chan)