from __future__ import annotations
import attrs
import pytest
from .. import abc as tabc
from ..lowlevel import Task
def test_instrument_implements_hook_methods() -> None:
    attrs = {'before_run': (), 'after_run': (), 'task_spawned': (Task,), 'task_scheduled': (Task,), 'before_task_step': (Task,), 'after_task_step': (Task,), 'task_exited': (Task,), 'before_io_wait': (3.3,), 'after_io_wait': (3.3,)}
    mayonnaise = tabc.Instrument()
    for method_name, args in attrs.items():
        assert hasattr(mayonnaise, method_name)
        method = getattr(mayonnaise, method_name)
        assert callable(method)
        method(*args)