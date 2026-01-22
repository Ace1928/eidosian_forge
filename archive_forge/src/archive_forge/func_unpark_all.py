from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
def unpark_all(self) -> list[Task]:
    """Unpark all parked tasks."""
    return self.unpark(count=len(self))