from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
def manyrows(self: ResultInternal[_R], num: Optional[int]) -> List[_R]:
    if num is None:
        real_result = self._real_result if self._real_result else cast('Result[Any]', self)
        num = real_result._yield_per
    rows: List[_InterimRowType[Any]] = self._fetchmany_impl(num)
    if make_row:
        rows = [make_row(row) for row in rows]
    if post_creational_filter:
        rows = [post_creational_filter(row) for row in rows]
    return rows