import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def prog_data_list_to_token_data_list(self, data: List[Tuple[int, int, int, float]]) -> Iterator[Tuple[int, int, int]]:
    for d in data:
        token_data = self.prog_data_to_token_data(*d)
        if token_data is not None:
            yield token_data