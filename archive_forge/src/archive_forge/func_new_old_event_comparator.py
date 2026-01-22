import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def new_old_event_comparator(event):
    if hasattr(event, 'start_us'):
        return event.start_us() * 1000
    if hasattr(event, 'start_time_ns'):
        return event.start_time_ns
    raise Exception('Unknown Event Type')