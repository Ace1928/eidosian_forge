from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def process_key_events_and_redraw(self):
    """
        Go through the recorded time stamps of the key press and release
        events and update the view accordingly.
        """
    self.process_keys_and_redraw_scheduled = False
    t = time.time()
    m = matrix.identity(self.raytracing_data.RF, 4)
    any_key = False
    for k, last_and_release in self.key_to_last_accounted_and_release_time.items():
        dT = None
        if last_and_release[0] is None:
            last_and_release[1] = None
        elif not last_and_release[1] is None and t - last_and_release[1] > _ignore_key_release_time_s:
            dT = last_and_release[1] - last_and_release[0]
            last_and_release[0] = None
            last_and_release[1] = None
        else:
            dT = t - last_and_release[0]
            last_and_release[0] = t
        if dT is not None:
            RF = m.base_ring()
            m = m * self.keymapping[k](RF(dT * self.navigation_dict['rotationVelocity'][1]), RF(dT * self.navigation_dict['translationVelocity'][1]))
            any_key = True
    if not any_key:
        return
    self.view_state = self.raytracing_data.update_view_state(self.view_state, m)
    self.redraw_if_initialized()
    self.schedule_process_key_events_and_redraw(_refresh_delay_ms)