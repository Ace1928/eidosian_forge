import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def compute_idle_time(self):
    """
        Computes idle time of the profile.
        """
    idle = False
    idle_start = 0
    idle_intervals: List[Interval] = []
    if self.queue_depth_list and self.events:
        idle_intervals += [Interval(self.events[0].start_time_ns, self.queue_depth_list[0].start), Interval(self.queue_depth_list[-1].end, self.events[-1].end_time_ns)]
    for data_point in self.queue_depth_list:
        if data_point.queue_depth == 0 and (not idle):
            idle_start = data_point.end
            idle = True
        if data_point.queue_depth > 0 and idle:
            idle_intervals.append(Interval(idle_start, data_point.start))
            idle = False
    event_list = [e.event for e in self.metrics.keys()]
    for event in event_list:
        self.metrics[EventKey(event)].idle_time_ns = EventKey(event).intervals_overlap(idle_intervals)