from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..iapi import strip_draw_info, strip_label_details
@property
def top_strips(self) -> Strips:
    return Strips([s for s in self if s.position == 'top'])