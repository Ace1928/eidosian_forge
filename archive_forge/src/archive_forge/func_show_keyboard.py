from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def show_keyboard(self):
    """
        Convenience function to show the keyboard in managed mode.
        """
    if self.keyboard_mode == 'managed':
        self._bind_keyboard()