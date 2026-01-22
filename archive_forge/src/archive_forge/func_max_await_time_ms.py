from __future__ import annotations
import copy
import warnings
from collections import deque
from typing import (
from bson import RE_TYPE, _convert_raw_document_lists_to_streams
from bson.code import Code
from bson.son import SON
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import (
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.lock import _create_lock
from pymongo.message import (
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _CollationIn, _DocumentOut, _DocumentType
def max_await_time_ms(self, max_await_time_ms: Optional[int]) -> Cursor[_DocumentType]:
    """Specifies a time limit for a getMore operation on a
        :attr:`~pymongo.cursor.CursorType.TAILABLE_AWAIT` cursor. For all other
        types of cursor max_await_time_ms is ignored.

        Raises :exc:`TypeError` if `max_await_time_ms` is not an integer or
        ``None``. Raises :exc:`~pymongo.errors.InvalidOperation` if this
        :class:`Cursor` has already been used.

        .. note:: `max_await_time_ms` requires server version **>= 3.2**

        :Parameters:
          - `max_await_time_ms`: the time limit after which the operation is
            aborted

        .. versionadded:: 3.2
        """
    if not isinstance(max_await_time_ms, int) and max_await_time_ms is not None:
        raise TypeError('max_await_time_ms must be an integer or None')
    self.__check_okay_to_chain()
    if self.__query_flags & CursorType.TAILABLE_AWAIT:
        self.__max_await_time_ms = max_await_time_ms
    return self