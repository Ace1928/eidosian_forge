from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_top_strip_height(pack: LayoutPack) -> float:
    """
    Height taken up by the top strips
    """
    if not pack.strip_text_x:
        return 0
    artists = [st.patch if st.patch.get_visible() else st for st in pack.strip_text_x if st.patch.position == 'top']
    return max_height(pack, artists)