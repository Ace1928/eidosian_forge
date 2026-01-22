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
def test_lsp_that_hooks_select_gives_good_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from .. import _io_windows
    from .._windows_cffi import CData, WSAIoctls, _handle

    def patched_get_underlying(sock: int | CData, *, which: int=WSAIoctls.SIO_BASE_HANDLE) -> CData:
        if hasattr(sock, 'fileno'):
            sock = sock.fileno()
        if which == WSAIoctls.SIO_BSP_HANDLE_SELECT:
            return _handle(sock + 1)
        else:
            return _handle(sock)
    monkeypatch.setattr(_io_windows, '_get_underlying_socket', patched_get_underlying)
    with pytest.raises(RuntimeError, match='SIO_BASE_HANDLE and SIO_BSP_HANDLE_SELECT differ'):
        _core.run(sleep, 0)