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
def scroll_to_element(self, element: WebElement) -> ActionChains:
    """If the element is outside the viewport, scrolls the bottom of the
        element to the bottom of the viewport.

        :Args:
         - element: Which element to scroll into the viewport.
        """
    self.w3c_actions.wheel_action.scroll(origin=element)
    return self