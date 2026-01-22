from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
@property
def edge_coordinates(self) -> EdgeCoordinates:
    return EdgeCoordinates(layout=self)