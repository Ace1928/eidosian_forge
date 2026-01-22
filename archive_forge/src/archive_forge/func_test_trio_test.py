from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING
import pytest
from trio.testing import RaisesGroup
from .. import _core, sleep, socket as tsocket
from .._core._tests.tutil import can_bind_ipv6
from .._highlevel_generic import StapledStream, aclose_forcefully
from .._highlevel_socket import SocketListener
from ..testing import *
from ..testing._check_streams import _assert_raises
from ..testing._memory_streams import _UnboundedByteQueue
def test_trio_test() -> None:

    async def busy_kitchen(*, mock_clock: object, autojump_clock: object) -> None:
        ...
    with pytest.raises(ValueError, match='^too many clocks spoil the broth!$'):
        trio_test(busy_kitchen)(mock_clock=MockClock(), autojump_clock=MockClock(autojump_threshold=0))