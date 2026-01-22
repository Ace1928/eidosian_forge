from __future__ import annotations
from ..utils import keys_to_typing
from .interaction import KEY
from .interaction import Interaction
from .key_input import KeyInput
from .pointer_input import PointerInput
from .wheel_input import WheelInput
def send_keys(self, text: str | list) -> KeyActions:
    if not isinstance(text, list):
        text = keys_to_typing(text)
    for letter in text:
        self.key_down(letter)
        self.key_up(letter)
    return self