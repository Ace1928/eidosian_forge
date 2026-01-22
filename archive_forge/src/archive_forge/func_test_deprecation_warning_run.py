from typing import Awaitable, Callable
import pytest
import trio
def test_deprecation_warning_run() -> None:

    async def foo() -> None:
        ...

    async def foo_nursery() -> None:
        async with trio.open_nursery():
            ...

    async def foo_loose_nursery() -> None:
        async with trio.open_nursery(strict_exception_groups=False):
            ...

    def helper(fun: Callable[..., Awaitable[None]], num: int) -> None:
        with pytest.warns(trio.TrioDeprecationWarning, match='strict_exception_groups=False') as record:
            trio.run(fun, strict_exception_groups=False)
        assert len(record) == num
    helper(foo, 1)
    helper(foo_nursery, 1)
    helper(foo_loose_nursery, 2)