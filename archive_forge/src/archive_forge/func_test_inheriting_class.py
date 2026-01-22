from __future__ import annotations
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import pytest
import dask
import dask.threaded
from dask.base import DaskMethodsMixin, dont_optimize, tokenize
from dask.context import globalmethod
from dask.delayed import Delayed, delayed
from dask.typing import (
def test_inheriting_class() -> None:
    inheriting: Inheriting = Inheriting(increment(2))
    assert isinstance(inheriting, Inheriting)