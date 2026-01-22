from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
def trace_alloc(self, ref: List[ray.ObjectRef], loc: str):
    ref = ref[0]
    if ref not in self.allocated:
        meta = ray.experimental.get_object_locations([ref])
        size_bytes = meta.get('object_size', 0)
        if not size_bytes:
            size_bytes = -1
            from ray import cloudpickle as pickle
            try:
                obj = ray.get(ref, timeout=5.0)
                size_bytes = len(pickle.dumps(obj))
            except Exception:
                print('[mem_tracing] ERROR getting size')
                size_bytes = -1
        print(f'[mem_tracing] Allocated {size_bytes} bytes at {loc}: {ref}')
        entry = {'size_bytes': size_bytes, 'loc': loc}
        self.allocated[ref] = entry
        self.cur_mem += size_bytes
        self.peak_mem = max(self.cur_mem, self.peak_mem)