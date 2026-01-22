import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def cry(out=None, sepchr='=', seplen=49):
    """Return stack-trace of all active threads.

    See Also:
        Taken from https://gist.github.com/737056.
    """
    import threading
    out = WhateverIO() if out is None else out
    P = partial(print, file=out)
    tmap = {t.ident: t for t in threading.enumerate()}
    sep = sepchr * seplen
    for tid, frame in sys._current_frames().items():
        thread = tmap.get(tid)
        if not thread:
            continue
        P(f'{thread.name}')
        P(sep)
        traceback.print_stack(frame, file=out)
        P(sep)
        P('LOCAL VARIABLES')
        P(sep)
        pprint(frame.f_locals, stream=out)
        P('\n')
    return out.getvalue()