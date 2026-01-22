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
def build_task(example_func, func=None, inputs=None, configs=None, cache='NoOpCache', deterministic=True, task_name='taskname'):
    ts = function_to_taskspec(example_func, lambda ds: [d['data_type'] is str for d in ds])
    ns = _NodeSpec(None, task_name, {}, {}, {})
    ts._node_spec = ns
    if func is not None:
        ts.func = func
    ts.deterministic = deterministic
    wfctx = WorkflowContext(cache=cache)
    t = _Task(ts, wfctx)
    if inputs is not None:
        for k, v in inputs.items():
            t.inputs[k].dependency = 1
            t.inputs[k]._cached = True
            t.inputs[k]._cached_value = v
    if configs is not None:
        for k, v in configs.items():
            t.configs[k].set(v)
    t.update_by_cache()
    return t