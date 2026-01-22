from __future__ import annotations
import collections.abc
import uuid
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from functools import lru_cache
from typing import (
import attrs
from attrs import define, field
from fontTools.misc.arrayTools import unionRect
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen, ControlBoundsPen
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import OBJECT_LIBS_KEY
from ufoLib2.typing import Drawable, GlyphSet, HasIdentifier
def unionBounds(bounds1: BoundingBox | None, bounds2: BoundingBox | None) -> BoundingBox | None:
    if bounds1 is None:
        return bounds2
    if bounds2 is None:
        return bounds1
    return BoundingBox(*unionRect(bounds1, bounds2))