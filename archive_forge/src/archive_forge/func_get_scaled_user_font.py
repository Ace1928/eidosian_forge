from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol, List
import pyglet
from pyglet.font import base
def get_scaled_user_font(font_base: UserDefinedMappingFont, size: int):
    """This function will return a new font that can scale it's size based off the original base font."""
    new_font = UserDefinedMappingFont(font_base.name, font_base.default_char, size, font_base.mappings, font_base.ascent, font_base.descent, font_base.bold, font_base.italic, font_base.stretch, font_base.dpi, font_base.locale)
    new_font.enable_scaling(font_base.size)
    return new_font