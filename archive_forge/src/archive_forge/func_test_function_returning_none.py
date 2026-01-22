import abc
import concurrent.futures
import contextlib
import inspect
import sys
import time
import traceback
from typing import List, Tuple
import pytest
import duet
import duet.impl as impl
def test_function_returning_none(self):
    side_effects = []

    async def func(value):
        value = await duet.completed_future(value * 2)
        value = await duet.completed_future(value * 3)
        side_effects.append(value)
    assert duet.run(func, 1) is None
    assert side_effects == [2 * 3]