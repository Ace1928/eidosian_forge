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
def test_failed_future(self):

    async def func(value):
        try:
            await duet.failed_future(Exception())
            return value * 2
        except Exception:
            return value * 3
    assert duet.run(func, 1) == 3