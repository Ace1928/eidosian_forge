from __future__ import annotations
import ctypes
import ctypes.util
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import Any, Callable, Generic, TypeVar
import outcome
def get_os_thread_name_func() -> Callable[[int | None, str], None] | None:

    def namefunc(setname: Callable[[int, bytes], int], ident: int | None, name: str) -> None:
        if ident is not None:
            setname(ident, _to_os_thread_name(name))

    def darwin_namefunc(setname: Callable[[bytes], int], ident: int | None, name: str) -> None:
        if ident is not None:
            setname(_to_os_thread_name(name))
    libpthread_path = ctypes.util.find_library('pthread')
    if not libpthread_path:
        libpthread_path = 'libc.so'
    try:
        libpthread = ctypes.CDLL(libpthread_path)
    except Exception:
        return None
    pthread_setname_np = getattr(libpthread, 'pthread_setname_np', None)
    if pthread_setname_np is None:
        return None
    pthread_setname_np.restype = ctypes.c_int
    if sys.platform == 'darwin':
        pthread_setname_np.argtypes = [ctypes.c_char_p]
        return partial(darwin_namefunc, pthread_setname_np)
    pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return partial(namefunc, pthread_setname_np)