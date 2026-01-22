import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def delta_to_wait_ids(self, delta_ms: float) -> Iterator[int]:

    def roundi(f: float):
        return ceil(f - 0.5)
    max_wait_ms = self.cfg.max_wait_time
    div = max_wait_ms / self.cfg.wait_events
    if delta_ms > max_wait_ms * 10:
        delta_ms = max_wait_ms * 10
    for _ in range(floor(delta_ms / max_wait_ms)):
        yield roundi(max_wait_ms / div)
    leftover_time_shift = roundi(delta_ms % max_wait_ms / div)
    if leftover_time_shift > 0:
        yield leftover_time_shift