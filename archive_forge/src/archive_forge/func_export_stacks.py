import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def export_stacks(self, path: str, metric: str='self_cpu_time_total'):
    self._check_finish()
    assert self.function_events is not None, 'Expected profiling results'
    assert self.with_stack, 'export_stacks() requires with_stack=True'
    return self.function_events.export_stacks(path, metric)