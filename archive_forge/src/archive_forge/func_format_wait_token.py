import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@lru_cache(maxsize=128)
def format_wait_token(self, wait: int) -> str:
    return f't{wait}'