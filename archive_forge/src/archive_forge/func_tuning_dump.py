from ctypes import *
from ctypes.util import find_library
import os
def tuning_dump(self, bank, prog, pitch):
    return fluid_synth_tuning_dump(self.synth, bank, prog, name.encode(), length(name), pitch)