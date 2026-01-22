from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas, SolidCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError, WidgetWarning
def padding_values(self, size: tuple[()] | tuple[int] | tuple[int, int], focus: bool) -> tuple[int, int]:
    """Return the number of columns to pad on the left and right.

        Override this method to define custom padding behaviour."""
    if self._width_type == WHSettings.CLIP:
        width, _ignore = self._original_widget.pack((), focus=focus)
        if not size:
            raise PaddingError('WHSettings.CLIP makes Padding FLOW-only widget')
        return calculate_left_right_padding(size[0], self._align_type, self._align_amount, WHSettings.CLIP, width, None, self.left, self.right)
    if self._width_type == WHSettings.PACK:
        if size:
            maxcol = size[0]
            maxwidth = max(maxcol - self.left - self.right, self.min_width or 0)
            width, _ignore = self._original_widget.pack((maxwidth,), focus=focus)
        else:
            width, _ignore = self._original_widget.pack((), focus=focus)
            maxcol = width + self.left + self.right
        return calculate_left_right_padding(maxcol, self._align_type, self._align_amount, WHSettings.GIVEN, width, self.min_width, self.left, self.right)
    if size:
        maxcol = size[0]
    elif self._width_type == WHSettings.GIVEN:
        maxcol = self._width_amount + self.left + self.right
    else:
        maxcol = max(self._original_widget.pack((), focus=focus)[0] * 100 // self._width_amount, self.min_width or 1) + self.left + self.right
    return calculate_left_right_padding(maxcol, self._align_type, self._align_amount, self._width_type, self._width_amount, self.min_width, self.left, self.right)