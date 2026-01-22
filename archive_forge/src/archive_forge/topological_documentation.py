from __future__ import annotations
from typing import Any
from typing import Collection
from typing import DefaultDict
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from .. import util
from ..exc import CircularDependencyError
sort the given list of items by dependency.

    'tuples' is a list of tuples representing a partial ordering.

    deterministic_order is no longer used, the order is now always
    deterministic given the order of "allitems".    the flag is there
    for backwards compatibility with Alembic.

    