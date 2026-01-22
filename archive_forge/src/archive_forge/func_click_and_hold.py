from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput
def click_and_hold(self, element: Optional[WebElement]=None, button=MouseButton.LEFT):
    if element:
        self.move_to(element)
    self.pointer_down(button=button)
    return self