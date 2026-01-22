from ctypes import *
from ctypes.util import find_library
import os
def sfont_select(self, chan, sfid):
    """Choose a SoundFont"""
    return fluid_synth_sfont_select(self.synth, chan, sfid)