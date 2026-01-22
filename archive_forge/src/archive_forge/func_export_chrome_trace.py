import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def export_chrome_trace(self, path):
    self._check_finish()
    assert self.function_events is not None
    return self.function_events.export_chrome_trace(path)