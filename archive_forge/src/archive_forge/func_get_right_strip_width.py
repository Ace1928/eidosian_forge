from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_right_strip_width(pack: LayoutPack) -> float:
    """
    Width taken up by the right strips
    """
    if not pack.strip_text_y:
        return 0
    artists = [st.patch if st.patch.get_visible() else st for st in pack.strip_text_y if st.patch.position == 'right']
    return max_width(pack, artists)