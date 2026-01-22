import typing
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement
from .input_device import InputDevice
from .interaction import POINTER
from .interaction import POINTER_KINDS
def create_pointer_up(self, button):
    self.add_action({'type': 'pointerUp', 'duration': 0, 'button': button})