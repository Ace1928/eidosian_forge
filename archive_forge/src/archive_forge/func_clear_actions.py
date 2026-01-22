from typing import List
from typing import Optional
from typing import Union
from selenium.webdriver.remote.command import Command
from . import interaction
from .key_actions import KeyActions
from .key_input import KeyInput
from .pointer_actions import PointerActions
from .pointer_input import PointerInput
from .wheel_actions import WheelActions
from .wheel_input import WheelInput
def clear_actions(self) -> None:
    """Clears actions that are already stored on the remote end."""
    self.driver.execute(Command.W3C_CLEAR_ACTIONS)