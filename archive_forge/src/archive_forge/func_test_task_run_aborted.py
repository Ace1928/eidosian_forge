import threading
import time
from collections import OrderedDict
from threading import RLock
from time import sleep
from typing import Any, Tuple
from adagio.exceptions import AbortedError
from adagio.instances import (NoOpCache, ParallelExecutionEngine, TaskContext,
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import InputSpec, OutputSpec, WorkflowSpec, _NodeSpec
from pytest import raises
from triad.exceptions import InvalidOperationError
from timeit import timeit
def test_task_run_aborted():
    ts = build_task(example_helper1, t3, inputs=dict(a=1), configs=dict(b='xx'))
    assert _State.CREATED == ts.state
    ts.ctx.abort()
    ts.run()
    assert _State.SKIPPED == ts.state
    assert ts.outputs['_0'].is_skipped
    assert ts.outputs['_1'].is_skipped
    ts = build_task(example_helper1, t4, inputs=dict(a=1), configs=dict(b='xx'))
    assert _State.CREATED == ts.state
    x = threading.Thread(target=ts.run)
    x.start()
    time.sleep(0.1)
    ts.ctx.abort()
    x.join()
    assert _State.ABORTED == ts.state
    assert ts.outputs['_0'].is_successful
    assert 3 == ts.outputs['_0'].value
    assert ts.outputs['_1'].is_failed
    assert isinstance(ts.outputs['_1'].exception, AbortedError)