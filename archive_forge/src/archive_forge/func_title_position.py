from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@cached_property
def title_position(self) -> SidePosition:
    if not (pos := self.theme.getp('legend_title_position')):
        pos = 'top' if self.is_vertical else 'left'
    return pos