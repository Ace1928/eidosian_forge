from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_get_stack_limit(self):
    d = {'x%d' % (i + 1): (inc, 'x%d' % i) for i in range(10000)}
    d['x0'] = 0
    assert self.get(d, 'x10000') == 10000