from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def get_rgb_values(self) -> tuple[int | None, int | None, int | None, int | None, int | None, int | None]:
    """
        Return (fg_red, fg_green, fg_blue, bg_red, bg_green, bg_blue) color
        components.  Each component is in the range 0-255.  Values are taken
        from the XTerm defaults and may not exactly match the user's terminal.

        If the foreground or background is 'default' then all their compenents
        will be returned as None.

        >>> AttrSpec('yellow', '#ccf', colors=88).get_rgb_values()
        (255, 255, 0, 205, 205, 255)
        >>> AttrSpec('default', 'g92').get_rgb_values()
        (None, None, None, 238, 238, 238)
        """
    if not (self.foreground_basic or self.foreground_high or self.foreground_true):
        vals = (None, None, None)
    elif self.colors == 88:
        if self.foreground_number >= 88:
            raise ValueError(f'Invalid AttrSpec _value: {self.foreground_number!r}')
        vals = _COLOR_VALUES_88[self.foreground_number]
    elif self.colors == 2 ** 24:
        h = f'{self.foreground_number:06x}'
        vals = tuple((int(x, 16) for x in (h[0:2], h[2:4], h[4:6])))
    else:
        vals = _COLOR_VALUES_256[self.foreground_number]
    if not (self.background_basic or self.background_high or self.background_true):
        return (*vals, None, None, None)
    if self.colors == 88:
        if self.background_number >= 88:
            raise ValueError(f'Invalid AttrSpec _value: {self.background_number!r}')
        return vals + _COLOR_VALUES_88[self.background_number]
    if self.colors == 2 ** 24:
        h = f'{self.background_number:06x}'
        return vals + tuple((int(x, 16) for x in (h[0:2], h[2:4], h[4:6])))
    return vals + _COLOR_VALUES_256[self.background_number]