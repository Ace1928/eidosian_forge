from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def tkAltButton1(self, event):
    for last, release in self.key_to_last_accounted_and_release_time.values():
        if last or release:
            return
    self.make_current()
    depth, width, height = self.read_depth_value(event.x, event.y)
    self.orbit_translation, self.orbit_inv_translation, self.orbit_speed = self.compute_translation_and_inverse_from_pick_point((width, height), (event.x, height - event.y), depth)
    self.last_mouse_pos = (event.x, event.y)
    self.view_state_when_pressed = self.view_state
    self.orbit_rotation = matrix.identity(self.raytracing_data.RF, 4)
    self.mouse_mode = 'orbit'