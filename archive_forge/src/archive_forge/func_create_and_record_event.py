from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
def create_and_record_event() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event