from ctypes import *
from ctypes.util import find_library
import os
def player_set_tempo(self, tempo_type, tempo):
    return fluid_player_set_tempo(self.player, tempo_type, tempo)