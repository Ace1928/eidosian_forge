from contextlib import contextmanager
import numpy as np
from_record_like = None
def sentry_contiguous(ary):
    core = array_core(ary)
    if not core.flags['C_CONTIGUOUS'] and (not core.flags['F_CONTIGUOUS']):
        raise ValueError(errmsg_contiguous_buffer)