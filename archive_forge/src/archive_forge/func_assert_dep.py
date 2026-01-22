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
def assert_dep(node, up, down):
    assert set(list(up)) == (up if isinstance(up, set) else set((t.name for t in wf.tasks[node].upstream)))
    assert set(list(down)) == (down if isinstance(down, set) else set((t.name for t in wf.tasks[node].downstream)))