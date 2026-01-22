from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils
def create_interrupt_this_thread_callback():
    """
    The idea here is returning a callback that when called will generate a KeyboardInterrupt
    in the thread that called this function.

    If this is the main thread, this means that it'll emulate a Ctrl+C (which may stop I/O
    and sleep operations).

    For other threads, this will call PyThreadState_SetAsyncExc to raise
    a KeyboardInterrupt before the next instruction (so, it won't really interrupt I/O or
    sleep operations).

    :return callable:
        Returns a callback that will interrupt the current thread (this may be called
        from an auxiliary thread).
    """
    tid = thread_get_ident()
    if is_current_thread_main_thread():
        main_thread = threading.current_thread()

        def raise_on_this_thread():
            pydev_log.debug('Callback to interrupt main thread.')
            pydevd_utils.interrupt_main_thread(main_thread)
    else:

        def raise_on_this_thread():
            if IS_CPYTHON:
                pydev_log.debug('Interrupt thread: %s', tid)
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt))
            else:
                pydev_log.debug('It is only possible to interrupt non-main threads in CPython.')
    return raise_on_this_thread