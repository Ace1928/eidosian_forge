import logging
import threading
import cupy
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.const import ENV
def get_stream_pool(device_idx):
    """Get the CUDA stream pool of a GPU device."""
    lock = threading.Lock()
    lock.acquire()
    if not _device_stream_pool_map:
        _init_stream_pool()
    lock.release()
    return _device_stream_pool_map[device_idx]