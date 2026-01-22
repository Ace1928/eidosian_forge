from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
def leak_report(self) -> str:
    output = StringIO()
    output.write('[mem_tracing] ===== Leaked objects =====\n')
    for ref in self.allocated:
        size_bytes = self.allocated[ref].get('size_bytes')
        loc = self.allocated[ref].get('loc')
        if ref in self.skip_dealloc:
            dealloc_loc = self.skip_dealloc[ref]
            output.write(f'[mem_tracing] Leaked object, created at {loc}, size {size_bytes}, skipped dealloc at {dealloc_loc}: {ref}\n')
        else:
            output.write(f'[mem_tracing] Leaked object, created at {loc}, size {size_bytes}: {ref}\n')
    output.write('[mem_tracing] ===== End leaked objects =====\n')
    output.write('[mem_tracing] ===== Freed objects =====\n')
    for ref in self.deallocated:
        size_bytes = self.deallocated[ref].get('size_bytes')
        loc = self.deallocated[ref].get('loc')
        dealloc_loc = self.deallocated[ref].get('dealloc_loc')
        output.write(f'[mem_tracing] Freed object from {loc} at {dealloc_loc}, size {size_bytes}: {ref}\n')
    output.write('[mem_tracing] ===== End freed objects =====\n')
    output.write(f'[mem_tracing] Peak size bytes {self.peak_mem}\n')
    output.write(f'[mem_tracing] Current size bytes {self.cur_mem}\n')
    return output.getvalue()