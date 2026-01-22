from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
@property
def background(self) -> str:
    """Return the background color."""
    if not (self.background_basic or self.background_high or self.background_true):
        return 'default'
    if self.background_basic:
        return _BASIC_COLORS[self.background_number]
    if self.__value & _HIGH_88_COLOR:
        return _color_desc_88(self.background_number)
    if self.colors == 2 ** 24:
        return _color_desc_true(self.background_number)
    return _color_desc_256(self.background_number)