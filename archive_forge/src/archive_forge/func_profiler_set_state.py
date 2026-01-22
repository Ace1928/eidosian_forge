import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle
def profiler_set_state(state='stop'):
    """Set up the profiler state to 'run' or 'stop' (Deprecated).

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    warnings.warn('profiler.profiler_set_state() is deprecated. Please use profiler.set_state() instead')
    set_state(state)