import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
def input_dtypes(event: _ProfilerEvent):
    assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
    return tuple((getattr(i, 'dtype', None) for i in event.extra_fields.inputs))