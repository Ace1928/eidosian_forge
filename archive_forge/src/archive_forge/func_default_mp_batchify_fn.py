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
def default_mp_batchify_fn(data):
    """Collate data into batch. Use shared memory for stacking."""
    if isinstance(data[0], nd.NDArray):
        empty_fn = _mx_np.empty if is_np_array() else nd.empty
        out = empty_fn((len(data),) + data[0].shape, dtype=data[0].dtype, ctx=context.Context('cpu_shared', 0))
        if is_np_array():
            return _mx_np.stack(data, out=out)
        else:
            return nd.stack(*data, out=out)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_mp_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        array_fn = _mx_np.array if is_np_array() else nd.array
        return array_fn(data, dtype=data.dtype, ctx=context.Context('cpu_shared', 0))