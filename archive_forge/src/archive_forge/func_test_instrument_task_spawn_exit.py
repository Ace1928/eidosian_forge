from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instrument_task_spawn_exit() -> None:
    record = []

    class SpawnExitRecorder(_abc.Instrument):

        def task_spawned(self, task: Task) -> None:
            record.append(('spawned', task))

        def task_exited(self, task: Task) -> None:
            record.append(('exited', task))

    async def main() -> Task:
        return _core.current_task()
    main_task = _core.run(main, instruments=[SpawnExitRecorder()])
    assert ('spawned', main_task) in record
    assert ('exited', main_task) in record