import pickle
import io
import sys
import signal
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
from . import sampler as _sampler
from ... import nd, context
from ...util import is_np_shape, is_np_array, set_np
from ... import numpy as _mx_np  # pylint: disable=reimported
def fetcher_loop_v1(data_queue, data_buffer, pin_memory=False, pin_device_id=0, data_buffer_lock=None):
    """Fetcher loop for fetching data from queue and put in reorder dict."""
    while True:
        idx, batch = data_queue.get()
        if idx is None:
            break
        if pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned(pin_device_id))
        else:
            batch = _as_in_context(batch, context.cpu())
        if data_buffer_lock is not None:
            with data_buffer_lock:
                data_buffer[idx] = batch
        else:
            data_buffer[idx] = batch