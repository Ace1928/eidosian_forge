import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sympy import Integer
from .. import metrics
from ..scheduler import SchedulerNode
from ..utils import ceildiv, Placeholder
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta
def jit_line(self):
    can_use_32bit = all((k.index_dtype == 'tl.int32' for k in self.sub_kernels))
    size_dtype = 'tl.int32' if can_use_32bit else 'tl.int64'
    _, _, signature = self.args.python_argdefs()
    triton_meta = {'signature': signature_to_meta(signature, size_dtype=size_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
    triton_meta['configs'] = [config_of(signature)]
    inductor_meta = {'kernel_name': str(Placeholder.DESCRIPTIVE_NAME)}
    return f'@foreach(num_warps={self.num_warps}, triton_meta={triton_meta!r}, inductor_meta={inductor_meta!r})\n' + '@triton.jit'