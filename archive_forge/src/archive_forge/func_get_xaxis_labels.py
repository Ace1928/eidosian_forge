from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_xaxis_labels(pack: LayoutPack, ax: Axes) -> Iterator[Text]:
    """
    Return all x-axis labels that will be shown
    """
    is_blank = pack.theme.T.is_blank
    major, minor = ([], [])
    if not is_blank('axis_text_x'):
        major = ax.xaxis.get_major_ticks()
        minor = ax.xaxis.get_minor_ticks()
    return (tick.label1 for tick in chain(major, minor) if _text_is_visible(tick.label1))