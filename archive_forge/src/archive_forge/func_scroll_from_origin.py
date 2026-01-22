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
def scroll_from_origin(self, scroll_origin: ScrollOrigin, delta_x: int, delta_y: int) -> ActionChains:
    """Scrolls by provided amount based on a provided origin. The scroll
        origin is either the center of an element or the upper left of the
        viewport plus any offsets. If the origin is an element, and the element
        is not in the viewport, the bottom of the element will first be
        scrolled to the bottom of the viewport.

        :Args:
         - origin: Where scroll originates (viewport or element center) plus provided offsets.
         - delta_x: Distance along X axis to scroll using the wheel. A negative value scrolls left.
         - delta_y: Distance along Y axis to scroll using the wheel. A negative value scrolls up.

         :Raises: If the origin with offset is outside the viewport.
          - MoveTargetOutOfBoundsException - If the origin with offset is outside the viewport.
        """
    if not isinstance(scroll_origin, ScrollOrigin):
        raise TypeError(f'Expected object of type ScrollOrigin, got: {type(scroll_origin)}')
    self.w3c_actions.wheel_action.scroll(origin=scroll_origin.origin, x=scroll_origin.x_offset, y=scroll_origin.y_offset, delta_x=delta_x, delta_y=delta_y)
    return self