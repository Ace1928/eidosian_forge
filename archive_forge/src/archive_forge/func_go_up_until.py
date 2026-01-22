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
def go_up_until(self, event: _ProfilerEvent, predicate):
    if not event:
        return None
    while event.parent and (not predicate(event)):
        event = event.parent
    return event