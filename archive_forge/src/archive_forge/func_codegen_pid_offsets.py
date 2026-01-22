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
@staticmethod
def codegen_pid_offsets(code, block_count, lower_bound, prefix):
    if block_count == 0:
        code.splice(f'{prefix}pid_offset = {prefix}pid')
    else:
        code.splice(f'{prefix}pid_offset = {prefix}pid - {lower_bound}')