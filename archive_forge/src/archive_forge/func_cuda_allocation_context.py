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
def cuda_allocation_context():
    snapshot = torch.cuda.memory._snapshot()
    addr_to_frame = {}
    for seg in snapshot['segments']:
        addr = seg['address']
        for blk in seg['blocks']:
            if blk['state'] == 'active_allocated':
                frames, real_size = _block_extra(blk)
                addr_to_frame[addr] = frames
            addr += blk['size']

    def object_context(obj):
        if is_cuda_tensor(obj):
            addr = obj.untyped_storage().data_ptr()
            frames = addr_to_frame.get(addr)
            if frames is not None:
                return '\n'.join(_frames_fmt(frames, full_filename=True))
        return None
    return object_context