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
def codegen_pid_range(self, code, x_elems):
    num_x_blocks = ceildiv(x_elems, self.get_block_size())
    upper_bound_x_pid = self.x_block_count + num_x_blocks
    lower_bound_x_pid = self.x_block_count
    if self.x_block_count == 0:
        cond = 'if'
    else:
        cond = 'elif'
    x_pid_bounds_check = f'xpid >= {lower_bound_x_pid} and xpid < {upper_bound_x_pid}'
    code.splice(f'{cond} {x_pid_bounds_check}:')
    with code.indent():
        ForeachKernel.codegen_pid_offsets(code, num_x_blocks, lower_bound_x_pid, 'x')
        self.x_block_count += num_x_blocks