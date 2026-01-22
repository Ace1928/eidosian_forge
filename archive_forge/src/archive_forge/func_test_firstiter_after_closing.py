from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def test_firstiter_after_closing() -> None:
    saved = []
    record = []

    async def funky_agen() -> AsyncGenerator[int, None]:
        try:
            yield 1
        except GeneratorExit:
            record.append('cleanup 1')
            raise
        try:
            yield 2
        finally:
            record.append('cleanup 2')
            await funky_agen().asend(None)

    async def async_main() -> None:
        aiter_ = funky_agen()
        saved.append(aiter_)
        assert await aiter_.asend(None) == 1
        assert await aiter_.asend(None) == 2
    _core.run(async_main)
    assert record == ['cleanup 2', 'cleanup 1']