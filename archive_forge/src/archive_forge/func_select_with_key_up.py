from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def select_with_key_up(self, keyboard, scancode, **kwargs):
    """(internal) Processes a key release. This must be called by the
        derived widget when a key that :meth:`select_with_key_down` returned
        True is released.

        The parameters are such that it could be bound directly to the
        on_key_up event of a keyboard.

        :Returns:
            bool, True if the key release was used, False otherwise.
        """
    if scancode[1] == 'shift':
        self._shift_down = False
    elif scancode[1] in ('ctrl', 'lctrl', 'rctrl'):
        self._ctrl_down = False
    else:
        try:
            self._key_list.remove(scancode[1])
            return True
        except ValueError:
            return False
    return True