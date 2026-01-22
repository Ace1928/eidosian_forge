from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def max_ylabels_bottom_protrusion(pack: LayoutPack, axes_loc: AxesLocation='all') -> float:
    """
    Return maximum height[inches] below the axes of y tick labels
    """

    def get_artist_bottom_y(a: Artist) -> float:
        xy = bbox_in_figure_space(a, pack.figure, pack.renderer).min
        return xy[1]
    extras = []
    for ax in filter_axes(pack.axs, axes_loc):
        ax_bottom = get_artist_bottom_y(ax)
        for label in get_yaxis_labels(pack, ax):
            label_bottom = get_artist_bottom_y(label)
            protrusion = abs(min(label_bottom - ax_bottom, 0))
            extras.append(protrusion)
    return max(extras) if len(extras) else 0