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
def test_workflow_run():
    """
       a   d       j
       |   |      /        b   e     |   k
       |   |     |   |
       c   |     aa  bb <----- sub start
        \\ /      |   |           f       _a  _b  |
         |       |  /|   |
         g       | | _c  |
        / \\      |  \\    |
       h   i     cc  dd  ee <----- sub end
                     |   |
                     l   m
    """
    s1 = SimpleSpec(['aa', 'bb'], ['cc', 'dd', 'ee'])
    s1.add('_a', example_helper_task1, '*aa')
    s1.add('_b', example_helper_task1, '*bb')
    s1.add('_c', example_helper_task1)
    s1.link('cc', '_a._0')
    s1.link('dd', '_b._0')
    s1.link('ee', 'bb')
    s = SimpleSpec()
    s.add('a', example_helper_task0)
    s.add('b', example_helper_task1)
    s.add('c', example_helper_task1)
    s.add('d', example_helper_task0)
    s.add('e', example_helper_task1)
    s.add('f', example_helper_task2, 'c', 'e')
    s.add('g', example_helper_task3)
    s.add('h', example_helper_task1, 'g._0')
    s.add('i', example_helper_task1, 'g._1')
    s.add('j', example_helper_task0)
    s.add('k', example_helper_task1)
    s.add_task('x', s1, {'aa': 'j._0', 'bb': 'k._0'})
    s.add('l', example_helper_task1, 'x.dd')
    s.add('m', example_helper_task1, 'x.ee')
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx.run(s, {})
    expected = {'a': 10, 'b': 11, 'c': 12, 'd': 10, 'e': 11, 'f': 23, 'h': 34, 'i': 14, 'j': 10, 'k': 11, '_a': 11, '_b': 12, '_c': 13, 'l': 13, 'm': 12}
    for k, v in expected.items():
        assert v == hooks.res[k]