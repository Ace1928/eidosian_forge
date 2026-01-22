import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def gc_callback(phase, info):
    nonlocal enabled
    if not enabled:
        return
    if phase == 'start':
        gc.set_debug(gc.DEBUG_SAVEALL)
    elif phase == 'stop':
        orig_trace = sys.getprofile()
        self_return = [False]

        def do_collect(*args, **kwargs):
            nonlocal enabled
            if not self_return[0]:
                self_return[0] = True
            else:
                sys.setprofile(orig_trace)
                enabled = False
                try:
                    if info['generation'] != 2:
                        gc.collect()
                    observer(gc.garbage)
                    gc.garbage.clear()
                    gc.set_debug(0)
                    before = torch.cuda.memory_allocated()
                    gc.collect()
                    after = torch.cuda.memory_allocated()
                    if before != after:
                        logger.warning('CUDA Memory changed during GC, %d bytes freed.', before - after)
                finally:
                    enabled = True
            if orig_trace is not None:
                return orig_trace(*args, **kwargs)
        sys.setprofile(do_collect)