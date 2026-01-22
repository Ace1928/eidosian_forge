from __future__ import annotations
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import create_autospec
import pytest
from ... import _core, sleep
from ...testing import wait_all_tasks_blocked
from .tutil import gc_collect_harder, restore_unraisablehook, slow
@restore_unraisablehook()
def test_forgot_to_register_with_iocp() -> None:
    with pipe_with_overlapped_read() as (write_fp, read_handle):
        with write_fp:
            write_fp.write(b'test\n')
        left_run_yet = False

        async def main() -> None:
            target = bytearray(1)
            try:
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(_core.readinto_overlapped, read_handle, target, name='xyz')
                    await wait_all_tasks_blocked()
                    nursery.cancel_scope.cancel()
            finally:
                assert left_run_yet
        with pytest.raises(_core.TrioInternalError) as exc_info:
            _core.run(main)
        left_run_yet = True
        assert 'Failed to cancel overlapped I/O in xyz ' in str(exc_info.value)
        assert 'forget to call register_with_iocp()?' in str(exc_info.value)
        del exc_info
        gc_collect_harder()