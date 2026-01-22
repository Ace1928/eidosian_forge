import sys
from cupy.cuda import memory_hook
def malloc_postprocess(self, **kwargs):
    msg = '{"hook":"%s","device_id":%d,"size":%d,"mem_size":%d,"mem_ptr":%d,"pmem_id":"%s"}'
    msg %= ('malloc', kwargs['device_id'], kwargs['size'], kwargs['mem_size'], kwargs['mem_ptr'], hex(kwargs['pmem_id']))
    self._print(msg)