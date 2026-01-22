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
def siblings_of(self, event: _ProfilerEvent):
    if event.parent:
        children = event.parent.children
    else:
        children = self.tid_root[event.start_tid]
    index = children.index(event)
    return (children[:index], children[index + 1:])