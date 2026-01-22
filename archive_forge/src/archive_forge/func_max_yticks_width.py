from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def max_yticks_width(pack: LayoutPack, axes_loc: AxesLocation='all') -> float:
    """
    Return maximum width[inches] of y ticks
    """
    widths = [tight_bbox_in_figure_space(tick.tick1line, pack.figure, pack.renderer).width for ax in filter_axes(pack.axs, axes_loc) for tick in get_yaxis_ticks(pack, ax)]
    return max(widths) if len(widths) else 0