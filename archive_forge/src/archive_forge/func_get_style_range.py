from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def get_style_range(self, attribute, start, end):
    """Get an attribute style over the given range.

        If the style varies over the range, `STYLE_INDETERMINATE` is returned.

        :Parameters:
            `attribute` : str
                Name of style attribute to query.
            `start` : int
                Starting character position.
            `end` : int
                Ending character position (exclusive).

        :return: The style set for the attribute over the given range, or
            `STYLE_INDETERMINATE` if more than one value is set.
        """
    iterable = self.get_style_runs(attribute)
    _, value_end, value = next(iterable.ranges(start, end))
    if value_end < end:
        return STYLE_INDETERMINATE
    else:
        return value