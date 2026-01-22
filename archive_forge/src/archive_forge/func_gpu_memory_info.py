import threading
import warnings
import ctypes
from .base import classproperty, with_metaclass, _MXClassPropertyMetaClass
from .base import _LIB
from .base import check_call
def gpu_memory_info(device_id=0):
    """Query CUDA for the free and total bytes of GPU global memory.

    Parameters
    ----------
    device_id : int, optional
        The device id of the GPU device.

    Raises
    ------
    Will raise an exception on any CUDA error.

    Returns
    -------
    (free, total) : (int, int)
    """
    free = ctypes.c_uint64()
    total = ctypes.c_uint64()
    dev_id = ctypes.c_int(device_id)
    check_call(_LIB.MXGetGPUMemoryInformation64(dev_id, ctypes.byref(free), ctypes.byref(total)))
    return (free.value, total.value)