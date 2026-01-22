from time import time
from kivy.config import Config
from kivy.vector import Vector
def find_double_tap(self, ref):
    """Find a double tap touch within self.touches.
        The touch must be not a previous double tap and the distance must be
        within the specified threshold. Additionally, the touch profiles
        must be the same kind of touch.
        """
    ref_button = None
    if 'button' in ref.profile:
        ref_button = ref.button
    for touchid in self.touches:
        if ref.uid == touchid:
            continue
        etype, touch = self.touches[touchid]
        if etype != 'end':
            continue
        if touch.is_double_tap:
            continue
        distance = Vector.distance(Vector(ref.sx, ref.sy), Vector(touch.osx, touch.osy))
        if distance > self.double_tap_distance:
            continue
        if touch.is_mouse_scrolling or ref.is_mouse_scrolling:
            continue
        touch_button = None
        if 'button' in touch.profile:
            touch_button = touch.button
        if touch_button != ref_button:
            continue
        touch.double_tap_distance = distance
        return touch