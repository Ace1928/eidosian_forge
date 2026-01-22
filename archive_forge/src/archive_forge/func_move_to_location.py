from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput
def move_to_location(self, x, y, width=None, height=None, pressure=None, tangential_pressure=None, tilt_x=None, tilt_y=None, twist=None, altitude_angle=None, azimuth_angle=None):
    self.source.create_pointer_move(origin='viewport', duration=self._duration, x=int(x), y=int(y), width=width, height=height, pressure=pressure, tangential_pressure=tangential_pressure, tilt_x=tilt_x, tilt_y=tilt_y, twist=twist, altitude_angle=altitude_angle, azimuth_angle=azimuth_angle)
    return self