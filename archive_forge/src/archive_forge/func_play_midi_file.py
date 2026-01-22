from ctypes import *
from ctypes.util import find_library
import os
def play_midi_file(self, filename):
    self.player = new_fluid_player(self.synth)
    if self.player == None:
        return FLUID_FAILED
    if self.custom_router_callback != None:
        fluid_player_set_playback_callback(self.player, self.custom_router_callback, self.synth)
    status = fluid_player_add(self.player, filename.encode())
    if status == FLUID_FAILED:
        return status
    status = fluid_player_play(self.player)
    return status