import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def wait_token_to_delta(self, token: str) -> float:
    return self.cfg.max_wait_time / self.cfg.wait_events * int(token[1:])