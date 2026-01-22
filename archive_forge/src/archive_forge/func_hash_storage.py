import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
def hash_storage(storage: torch.UntypedStorage, *, stable_hash: bool=False) -> str:
    import torch._dynamo
    from torch._dynamo.utils import is_compile_supported
    device_type = storage.device.type
    if stable_hash or not is_compile_supported(device_type):
        cpu_storage = storage.cpu()
        buf = (ctypes.c_byte * cpu_storage.nbytes()).from_address(cpu_storage.data_ptr())
        sha1 = hashlib.sha1()
        sha1.update(buf)
        return sha1.hexdigest()
    if device_type == 'cpu':
        generator = default_generator
    elif device_type == 'cuda':
        import torch.cuda
        generator = torch.cuda.default_generators[storage.device.index]
    else:
        raise AssertionError(f'unhandled device type {device_type}')
    state = generator.get_state()
    try:
        generator.manual_seed(0)
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
        pad = -x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), 'constant', 0)
        x = x.view(torch.int32)
        ITER = 5
        cs = [hash_storage_kernel(x).item() for _ in range(ITER)]
        return struct.pack('>' + 'i' * ITER, *cs).hex()
    finally:
        generator.set_state(state)