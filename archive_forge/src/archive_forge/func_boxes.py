from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@property
def boxes(self) -> list[FlexibleAnchoredOffsetbox]:
    """
        Return list of all AnchoredOffsetboxes for the legends
        """
    lrtb = (l.box for l in (self.left, self.right, self.top, self.bottom) if l)
    inside = (l.box for l in self.inside)
    return list(itertools.chain([*lrtb, *inside]))