from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
@classmethod
def get_all_timings(cls) -> str:
    """Returns the statistics of all the timings"""
    return cls.get_common_timings(cls.all_event_recorders, 'All timings')