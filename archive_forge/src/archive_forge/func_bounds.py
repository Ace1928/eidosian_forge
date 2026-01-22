from __future__ import annotations
import warnings
from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, overload
from attrs import define, field
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from ufoLib2.objects.misc import BoundingBox, getBounds, getControlBounds
from ufoLib2.objects.point import Point
from ufoLib2.serde import serde
from ufoLib2.typing import GlyphSet
@property
def bounds(self) -> BoundingBox | None:
    """Returns the (xMin, yMin, xMax, yMax) bounding box of the glyph,
        taking the actual contours into account.

        |defcon_compat|
        """
    return self.getBounds()