from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def tkButtonMotion1(self, event):
    if self.mouse_mode == 'orbit':
        delta_x = event.x - self.last_mouse_pos[0]
        delta_y = event.y - self.last_mouse_pos[1]
        RF = self.raytracing_data.RF
        angle_x = RF(delta_x * self.orbit_speed * 0.01)
        angle_y = RF(delta_y * self.orbit_speed * 0.01)
        m = O13_y_rotation(angle_x) * O13_x_rotation(angle_y)
        self.orbit_rotation = self.orbit_rotation * m
        self.view_state = self.raytracing_data.update_view_state(self.view_state_when_pressed, self.orbit_translation * self.orbit_rotation * self.orbit_inv_translation)
        self.last_mouse_pos = (event.x, event.y)
    elif self.mouse_mode == 'move':
        RF = self.raytracing_data.RF
        delta_x = RF(event.x - self.mouse_pos_when_pressed[0])
        delta_y = RF(event.y - self.mouse_pos_when_pressed[1])
        amt = (delta_x ** 2 + delta_y ** 2).sqrt()
        if amt == 0:
            self.view_state = self.view_state_when_pressed
        else:
            m = unit_3_vector_and_distance_to_O13_hyperbolic_translation([-delta_x / amt, delta_y / amt, RF(0)], amt * RF(0.01))
            self.view_state = self.raytracing_data.update_view_state(self.view_state_when_pressed, m)
    elif self.mouse_mode == 'rotate':
        RF = self.raytracing_data.RF
        delta_x = event.x - self.mouse_pos_when_pressed[0]
        delta_y = event.y - self.mouse_pos_when_pressed[1]
        angle_x = RF(-delta_x * 0.01)
        angle_y = RF(-delta_y * 0.01)
        m = O13_y_rotation(angle_x) * O13_x_rotation(angle_y)
        self.view_state = self.raytracing_data.update_view_state(self.view_state, m)
        self.mouse_pos_when_pressed = (event.x, event.y)
    else:
        return
    self.redraw_if_initialized()