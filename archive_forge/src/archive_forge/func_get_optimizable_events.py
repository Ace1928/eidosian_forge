import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def get_optimizable_events(self, length: int=1, print_enable: bool=True):
    event_list = self.rank_events(length)
    if not print_enable:
        return event_list
    output = 'Optimizable events:\n' if event_list else 'No events to optimize\n'
    output += '\n'.join([f'{'-' * 80}\nEvent:                {event}\nSource code location: {source_code_location(event.event)}\nPercentage idle time: {self.metrics[event].fraction_idle_time * 100:.2f}%\n{'-' * 80}' for event in event_list])
    if print_enable:
        print(output)
    return event_list