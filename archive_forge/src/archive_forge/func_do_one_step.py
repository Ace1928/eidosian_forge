import collections
import queue
import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper
def do_one_step():
    try:
        r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
    except queue.Empty:
        return
    idx, data = r
    if not done_event.is_set() and (not isinstance(data, ExceptionWrapper)):
        try:
            data = pin_memory(data, device)
        except Exception:
            data = ExceptionWrapper(where=f'in pin memory thread for device {device_id}')
        r = (idx, data)
    while not done_event.is_set():
        try:
            out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
            break
        except queue.Full:
            continue