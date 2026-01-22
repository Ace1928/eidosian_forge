from __future__ import annotations
import logging # isort:skip
import numbers
from datetime import date, datetime, timezone
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.descriptors import UnsetValueError
from ...core.property.singletons import Undefined
from ...core.validation import error
from ...core.validation.errors import EQUAL_SLIDER_START_END
from ..formatters import TickFormatter
from .widget import Widget
@property
def value_as_datetime(self) -> tuple[datetime, datetime] | None:
    """ Convenience property to retrieve the value tuple as a tuple of
        datetime objects.
        """
    if self.value is None:
        return None
    v1, v2 = self.value
    if isinstance(v1, numbers.Number):
        d1 = datetime.fromtimestamp(v1 / 1000, tz=timezone.utc)
    else:
        d1 = v1
    if isinstance(v2, numbers.Number):
        d2 = datetime.fromtimestamp(v2 / 1000, tz=timezone.utc)
    else:
        d2 = v2
    return (d1, d2)