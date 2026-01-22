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

       a   d       j
       |   |      /        b   e     |   k
       |   |     |   |
       c   |     aa  bb <----- sub start
        \ /      |   |           f       _a  _b  |
         |       |  /|   |
         g       | | _c  |
        / \      |  \    |
       h   i     cc  dd  ee <----- sub end
                     |   |
                     l   m
    