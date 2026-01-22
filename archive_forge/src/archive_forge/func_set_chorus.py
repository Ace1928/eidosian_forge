from ctypes import *
from ctypes.util import find_library
import os
def set_chorus(self, nr=-1, level=-1.0, speed=-1.0, depth=-1.0, type=-1):
    """
        nr Chorus voice count (0-99, CPU time consumption proportional to this value)
        level Chorus level (0.0-10.0)
        speed Chorus speed in Hz (0.29-5.0)
        depth_ms Chorus depth (max value depends on synth sample rate, 0.0-21.0 is safe for sample rate values up to 96KHz)
        type Chorus waveform type (0=sine, 1=triangle)
        """
    if fluid_synth_set_chorus is not None:
        return fluid_synth_set_chorus(self.synth, nr, level, speed, depth, type)
    else:
        set = 0
        if nr >= 0:
            set += 1
        if level >= 0:
            set += 2
        if speed >= 0:
            set += 4
        if depth >= 0:
            set += 8
        if type >= 0:
            set += 16
        return fluid_synth_set_chorus_full(self.synth, set, nr, level, speed, depth, type)