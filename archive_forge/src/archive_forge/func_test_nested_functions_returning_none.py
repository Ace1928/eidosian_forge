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
def test_nested_functions_returning_none(self):
    side_effects = []

    async def func(value):
        value2 = await sub_func(value * 2)
        return (value * 3, value2)

    async def sub_func(value):
        value = await duet.completed_future(value * 5)
        value = await duet.completed_future(value * 7)
        side_effects.append(value)
    assert duet.run(func, 1) == (3, None)
    assert side_effects == [2 * 5 * 7]