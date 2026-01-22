from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
def get_focus_widgets(self) -> list[Widget]:
    """
        Return the .focus values starting from this container
        and proceeding along each child widget until reaching a leaf
        (non-container) widget.

        Note that the list does not contain the topmost container widget
        (i.e., on which this method is called), but does include the
        lowest leaf widget.
        """
    out = []
    w = self
    while True:
        w = w.base_widget.focus
        if w is None:
            return out
        out.append(w)