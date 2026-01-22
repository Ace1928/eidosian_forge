from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def setup_keymapping(self, keyboard='QWERTY'):
    self.keymapping = _keymappings[keyboard]
    self.key_to_last_accounted_and_release_time = {k: [None, None] for k in self.keymapping}