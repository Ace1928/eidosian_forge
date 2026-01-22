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
def matched_events(self):
    if self.skip:
        return []
    matched_events = []
    for event in self.eventTreeTraversal():
        if self.match(event):
            matched_events.append(event)
    return matched_events