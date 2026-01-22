from typing import Awaitable, Callable
import pytest
import trio
def test_deprecation_warning_start_guest_run() -> None:
    from .._core._tests.test_guest_mode import trivial_guest_run

    async def trio_return(in_host: object) -> str:
        await trio.lowlevel.checkpoint()
        return 'ok'
    with pytest.warns(trio.TrioDeprecationWarning, match='strict_exception_groups=False') as record:
        trivial_guest_run(trio_return, strict_exception_groups=False)
    assert len(record) == 1