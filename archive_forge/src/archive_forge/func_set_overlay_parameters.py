from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasOverlay, CompositeCanvas
from urwid.split_repr import remove_defaults
from .constants import (
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .padding import calculate_left_right_padding
from .widget import Widget, WidgetError, WidgetWarning
def set_overlay_parameters(self, align: Literal['left', 'center', 'right'] | Align | tuple[Literal['relative', 'fixed left', 'fixed right', WHSettings.RELATIVE], int], width: Literal['pack', WHSettings.PACK] | int | tuple[Literal['relative', WHSettings.RELATIVE], int] | None, valign: Literal['top', 'middle', 'bottom'] | VAlign | tuple[Literal['relative', 'fixed top', 'fixed bottom', WHSettings.RELATIVE], int], height: Literal['pack', WHSettings.PACK] | int | tuple[Literal['relative', WHSettings.RELATIVE], int] | None, min_width: int | None=None, min_height: int | None=None, left: int=0, right: int=0, top: int=0, bottom: int=0) -> None:
    """
        Adjust the overlay size and position parameters.

        See :class:`__init__() <Overlay>` for a description of the parameters.
        """
    if isinstance(align, tuple):
        if align[0] == 'fixed left':
            left = align[1]
            normalized_align = Align.LEFT
        elif align[0] == 'fixed right':
            right = align[1]
            normalized_align = Align.RIGHT
        else:
            normalized_align = align
    else:
        normalized_align = Align(align)
    if isinstance(width, tuple):
        if width[0] == 'fixed left':
            left = width[1]
            width = RELATIVE_100
        elif width[0] == 'fixed right':
            right = width[1]
            width = RELATIVE_100
    if isinstance(valign, tuple):
        if valign[0] == 'fixed top':
            top = valign[1]
            normalized_valign = VAlign.TOP
        elif valign[0] == 'fixed bottom':
            bottom = valign[1]
            normalized_valign = VAlign.BOTTOM
        else:
            normalized_valign = valign
    elif not isinstance(valign, (VAlign, str)):
        raise OverlayError(f'invalid valign: {valign!r}')
    else:
        normalized_valign = VAlign(valign)
    if isinstance(height, tuple):
        if height[0] == 'fixed bottom':
            bottom = height[1]
            height = RELATIVE_100
        elif height[0] == 'fixed top':
            top = height[1]
            height = RELATIVE_100
    if width is None:
        width = WHSettings.PACK
    if height is None:
        height = WHSettings.PACK
    align_type, align_amount = normalize_align(normalized_align, OverlayError)
    width_type, width_amount = normalize_width(width, OverlayError)
    valign_type, valign_amount = normalize_valign(normalized_valign, OverlayError)
    height_type, height_amount = normalize_height(height, OverlayError)
    if height_type in {WHSettings.GIVEN, WHSettings.PACK}:
        min_height = None
    self.contents[1] = (self.top_w, self.options(align_type, align_amount, width_type, width_amount, valign_type, valign_amount, height_type, height_amount, min_width, min_height, left, right, top, bottom))