import signal
import sys
import types
from typing import Any, TypeVar
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import _core
from .._core._tests.tutil import (
from .._util import (
from ..testing import wait_all_tasks_blocked
def test_no_public_constructor_metaclass() -> None:
    """The NoPublicConstructor metaclass prevents calling the constructor directly."""

    class SpecialClass(metaclass=NoPublicConstructor):

        def __init__(self, a: int, b: float) -> None:
            """Check arguments can be passed to __init__."""
            assert a == 8
            assert b == 3.14
    with pytest.raises(TypeError):
        SpecialClass(8, 3.14)
    assert isinstance(SpecialClass._create(8, b=3.14), SpecialClass)