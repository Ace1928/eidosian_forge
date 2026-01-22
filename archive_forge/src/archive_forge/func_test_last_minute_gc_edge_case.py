from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
@restore_unraisablehook()
def test_last_minute_gc_edge_case() -> None:
    saved: list[AsyncGenerator[int, None]] = []
    record = []
    needs_retry = True

    async def agen() -> AsyncGenerator[int, None]:
        try:
            yield 1
        finally:
            record.append('cleaned up')

    def collect_at_opportune_moment(token: _core._entry_queue.TrioToken) -> None:
        runner = _core._run.GLOBAL_RUN_CONTEXT.runner
        assert runner.system_nursery is not None
        if runner.system_nursery._closed and isinstance(runner.asyncgens.alive, weakref.WeakSet):
            saved.clear()
            record.append('final collection')
            gc_collect_harder()
            record.append('done')
        else:
            try:
                token.run_sync_soon(collect_at_opportune_moment, token)
            except _core.RunFinishedError:
                nonlocal needs_retry
                needs_retry = True

    async def async_main() -> None:
        token = _core.current_trio_token()
        token.run_sync_soon(collect_at_opportune_moment, token)
        saved.append(agen())
        await saved[-1].asend(None)
    for _attempt in range(50):
        needs_retry = False
        del record[:]
        del saved[:]
        _core.run(async_main)
        if needs_retry:
            assert record == ['cleaned up']
        else:
            assert record == ['final collection', 'done', 'cleaned up']
            break
    else:
        pytest.fail(f"Didn't manage to hit the trailing_finalizer_asyncgens case despite trying {_attempt} times")