import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def note_token_to_data(self, token: str) -> Tuple[int, int, int]:
    instr_str, note_str, velocity_str = token.strip().split(':')
    instr_bin = self.cfg._short_instrument_names_str_to_int[instr_str]
    note = int(note_str, base=16)
    velocity = self.bin_to_velocity(int(velocity_str, base=16))
    return (instr_bin, note, velocity)