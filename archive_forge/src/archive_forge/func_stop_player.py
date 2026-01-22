from kivy.app import App
from kivy.uix.videoplayer import VideoPlayer
from kivy.clock import Clock
import os
import time
def stop_player(self, *args):
    if time.perf_counter() - self.start_t > 10:
        assert self.player.duration > 0
        assert self.player.position > 0
        self.stop()
    elif self.player.position > 0 and self.player.duration > 0:
        self.stop()