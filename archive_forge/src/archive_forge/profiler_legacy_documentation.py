import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
Return a tuple for correlating start and end records in `_parse_legacy_records`.