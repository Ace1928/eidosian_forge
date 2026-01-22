import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def velocity_to_bin(self, velocity: float) -> int:
    velocity = max(0, min(velocity, self.cfg.velocity_events - 1))
    if self.cfg.velocity_bins_override:
        for i, v in enumerate(self.cfg.velocity_bins_override):
            if velocity <= v:
                return i
        return 0
    binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
    if self.cfg.velocity_exp == 1.0:
        return ceil(velocity / binsize)
    else:
        return ceil(self.cfg.velocity_events * ((self.cfg.velocity_exp ** (velocity / self.cfg.velocity_events) - 1.0) / (self.cfg.velocity_exp - 1.0)) / binsize)