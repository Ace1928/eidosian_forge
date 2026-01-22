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
def benchmark_summary(self, events: List[_ProfilerEvent]):

    def format_time(time_ns: int):
        unit_lst = ['ns', 'us', 'ms']
        for unit in unit_lst:
            if time_ns < 1000:
                return f'{time_ns:.2f} {unit}'
            time_ns //= 1000
        return f'{time_ns:.2f} s'
    assert hasattr(self, 'benchmark'), 'Please implement benchmark()'
    shapes_factor_map = self.benchmark(events)
    original_time = sum((event.duration_time_ns for event in events))
    new_time = sum((shapes_factor_map[input_shapes(event)] * event.duration_time_ns for event in events))
    return f'{self.name}: {len(events)} events matched. Total Estimated Speedup: {format_time(original_time - new_time)} ({round(original_time / new_time, 2)}X)'