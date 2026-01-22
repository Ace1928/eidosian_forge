from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..iapi import strip_draw_info, strip_label_details
@staticmethod
def from_facet(facet: facet) -> Strips:
    new = Strips()
    new.facet = facet
    new.setup()
    return new