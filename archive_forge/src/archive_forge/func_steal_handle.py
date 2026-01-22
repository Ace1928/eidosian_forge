from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def steal_handle(source_pid, handle):
    """Steal a handle from process identified by source_pid."""
    source_process_handle = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, source_pid)
    try:
        return _winapi.DuplicateHandle(source_process_handle, handle, _winapi.GetCurrentProcess(), 0, False, _winapi.DUPLICATE_SAME_ACCESS | _winapi.DUPLICATE_CLOSE_SOURCE)
    finally:
        _winapi.CloseHandle(source_process_handle)