import typing
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement
from .input_device import InputDevice
from .interaction import POINTER
from .interaction import POINTER_KINDS
def create_pause(self, pause_duration: float) -> None:
    self.add_action({'type': 'pause', 'duration': int(pause_duration * 1000)})