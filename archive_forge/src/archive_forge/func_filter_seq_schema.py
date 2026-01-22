from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def filter_seq_schema(*, include: Set[int] | None=None, exclude: Set[int] | None=None) -> IncExSeqSerSchema:
    return _dict_not_none(type='include-exclude-sequence', include=include, exclude=exclude)