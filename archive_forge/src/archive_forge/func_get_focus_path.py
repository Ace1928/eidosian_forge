from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
def get_focus_path(self) -> list[int | str]:
    """
        Return the .focus_position values starting from this container
        and proceeding along each child widget until reaching a leaf
        (non-container) widget.
        """
    out = []
    w = self
    while True:
        try:
            p = w.focus_position
        except IndexError:
            return out
        out.append(p)
        w = w.focus.base_widget