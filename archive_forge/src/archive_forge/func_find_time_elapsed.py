from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
def find_time_elapsed(self) -> float:
    if self.end_event is None:
        raise Exception(f'stopEvent was not called for event with name {self.event_name}')
    self.end_event.synchronize()
    return self.start_event.elapsed_time(self.end_event)