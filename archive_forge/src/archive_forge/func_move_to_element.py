from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from .actions.action_builder import ActionBuilder
from .actions.key_input import KeyInput
from .actions.pointer_input import PointerInput
from .actions.wheel_input import ScrollOrigin
from .actions.wheel_input import WheelInput
from .utils import keys_to_typing
def move_to_element(self, to_element: WebElement) -> ActionChains:
    """Moving the mouse to the middle of an element.

        :Args:
         - to_element: The WebElement to move to.
        """
    self.w3c_actions.pointer_action.move_to(to_element)
    self.w3c_actions.key_action.pause()
    return self