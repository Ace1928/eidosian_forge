import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def on_joybutton_release(self, joystick, button):
    """A button on the joystick was released.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose button was released.
            `button` : int
                The index (in `button_controls`) of the button that was released.
        """