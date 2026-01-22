from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
def legend_aesthetics(self, layer):
    """
        Return the aesthetics that contribute to the legend

        Parameters
        ----------
        layer : Layer
            Layer whose legend is to be drawn

        Returns
        -------
        matched : list
            List of the names of the aethetics that contribute
            to the legend.
        """
    l = layer
    legend_ae = set(self.key.columns) - {'label'}
    all_ae = l.mapping.keys() | (self.plot_mapping if l.inherit_aes else set()) | l.stat.DEFAULT_AES.keys()
    geom_ae = l.geom.REQUIRED_AES | l.geom.DEFAULT_AES.keys()
    matched = all_ae & geom_ae & legend_ae
    matched = list(matched - set(l.geom.aes_params))
    return matched